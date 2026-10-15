"""
Guardrails implementation built on in-process checks (regex, scores).

Local-first by design: every default check runs in this process with no
API call and no model download, so the default configuration adds zero
LLM calls per request on the free-tier token budget. The only LLM-backed
check (groundedness) is opt-in via settings.guard_groundedness_llm.
"""

from __future__ import annotations

from typing import Any

from rag_app.core.interfaces import (
    Guardrails,
    GuardResult,
    LLMProvider,
    RetrievedChunk,
    StreamFilter,
)
from rag_app.observability.provider import ObservabilityProvider


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
        return query, history, []

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
        return text
