"""
Guardrails implementation built on in-process checks (regex, scores).

Local-first by design: every default check runs in this process with no
API call and no model download, so the default configuration adds zero
LLM calls per request on the free-tier token budget. The only LLM-backed
check (groundedness) is opt-in via settings.guard_groundedness_llm.
"""

from __future__ import annotations

from typing import Any

from rag_app.config.settings import settings
from rag_app.core.interfaces import (
    Guardrails,
    GuardResult,
    LLMProvider,
    RetrievedChunk,
    StreamFilter,
    first_block,
)
from rag_app.guardrails import patterns
from rag_app.observability.provider import ObservabilityProvider

REFUSAL_INPUT = (
    "I can't help with that request. I can answer questions about the ingested documents."
)

# How many of the most recent stored messages are screened. Generation only
# ever sees the last few turns (routes/chat.py _format_history), so older
# messages can't reach the prompt and aren't worth re-scanning per request;
# twice that window leaves room for excluded turns.
HISTORY_SCAN = 12


class _PassThroughFilter(StreamFilter):
    def __init__(self) -> None:
        self.results: list[GuardResult] = []

    def feed(self, token: str) -> str:
        return token

    def flush(self) -> str:
        return ""


class RegexGuardrails(Guardrails):
    """Guardrails backed by in-process checks.

    Settings are read at call time rather than captured in __init__, so the
    lru_cache'd singleton picks up a monkeypatched setting in tests.
    """

    def __init__(self, obs: ObservabilityProvider, llm: LLMProvider | None = None) -> None:
        self._obs = obs
        self._llm = llm

    async def check_input(
        self, query: str, history: list[dict[str, Any]], request_id: str = ""
    ) -> tuple[str, list[dict[str, Any]], list[GuardResult]]:
        if not settings.guardrails_enabled:
            return query, history, []
        threshold = settings.guard_injection_block_score
        results: list[GuardResult] = []

        score, hits = patterns.injection_score(query)
        if score > 0:
            blocked = score >= threshold
            results.append(
                GuardResult(
                    "block" if blocked else "flag",
                    "input.injection",
                    f"matched: {', '.join(hits)}",
                    score,
                    REFUSAL_INPUT if blocked else "",
                )
            )

        kept, dropped = _screen_history(history[-HISTORY_SCAN:], threshold)
        if dropped:
            results.append(
                GuardResult(
                    "flag", "input.history", f"{dropped} earlier message(s) excluded from the prompt"
                )
            )

        # Split payloads: a phrase spread over several turns ("remember the
        # words 'ignore all'" ... "now apply them to previous instructions")
        # only shows up when the turns are read together. Only phrases that
        # match the joined text but no single message count -- two separate
        # flagged mentions in different turns are not a split payload.
        recent = [m["content"] for m in kept if m.get("role") == "user"][-HISTORY_SCAN // 2 :]
        if recent and not first_block(results):
            joined_score, joined_hits = patterns.injection_score(" ".join([*recent, query]))
            single_hits = set(hits).union(*(patterns.injection_score(t)[1] for t in recent))
            spanning = [h for h in joined_hits if h not in single_hits and h != "obfuscation"]
            if spanning and joined_score >= threshold:
                results.append(
                    GuardResult(
                        "block",
                        "input.multi_turn",
                        f"matched across turns: {', '.join(spanning)}",
                        joined_score,
                        REFUSAL_INPUT,
                    )
                )

        # PII in the question is allowed through (a user may legitimately ask
        # about a contact listed in a document) but flagged, which keeps the
        # answer out of the shared cache. Secrets are never needed to answer
        # a question about the documents, so they're masked before the query
        # reaches retrieval, the LLM, or a log line.
        pii = patterns.find_pii(query)
        if pii:
            results.append(GuardResult("flag", "input.pii", f"found: {', '.join(pii).lower()}"))
        secrets_found: list[str] = []
        query = patterns.mask_secrets(query, secrets_found)
        if secrets_found:
            results.append(
                GuardResult("flag", "input.secret", f"masked {len(secrets_found)} secret(s)")
            )

        self._log("input", request_id, results)
        return query, kept, results

    def check_context(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        reranked: bool,
        has_history: bool,
        request_id: str = "",
    ) -> tuple[list[RetrievedChunk], list[GuardResult]]:
        return chunks, []

    def scan_chunk(self, text: str) -> tuple[str | None, list[str]]:
        return text, []

    def stream_filter(self, request_id: str = "") -> StreamFilter:
        return _PassThroughFilter()

    async def review_output(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        query: str,
        request_id: str = "",
    ) -> list[GuardResult]:
        return []

    def redact(self, text: str) -> str:
        return patterns.redact(text)

    def _log(self, stage: str, request_id: str, results: list[GuardResult]) -> None:
        """One event per stage per request, pass included, so a request's
        guard path can be followed end to end by request_id."""
        blocked = first_block(results)
        attrs = {
            "request_id": request_id,
            "stage": stage,
            "outcome": blocked.action if blocked else ("flag" if results else "pass"),
            "results": [
                {"check": r.check, "action": r.action, "reason": r.reason, "score": r.score}
                for r in results
            ],
        }
        (self._obs.log_warning if blocked else self._obs.log_event)("guard.decision", attrs)


def _screen_history(
    history: list[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], int]:
    """Drop stored turns that must not be fed back into generation: turns
    already refused (tagged with a blocking guardrail result when stored),
    turns that score as injection now (chats from before guardrails, or a
    lowered threshold), and the assistant reply to any excluded user turn.
    Returns (kept messages, number dropped)."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    reply_to_dropped = False
    for m in history:
        bad = (m.get("guardrail") or {}).get("action") == "block" or (
            patterns.injection_score(m.get("content", ""))[0] >= threshold
        )
        if m.get("role") == "user":
            reply_to_dropped = bad
        elif reply_to_dropped:
            bad, reply_to_dropped = True, False
        if bad:
            dropped += 1
        else:
            kept.append(m)
    return kept, dropped
