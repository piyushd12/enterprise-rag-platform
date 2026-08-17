"""
LLM Router — tries the primary provider (Groq) first, falls back to
OpenRouter on any error (rate limit, timeout, provider error).

Logs every fallback event to Logfire via the ObservabilityProvider.
The provider that actually served the request is recorded in the
LLMResponse.provider field and propagated as LangSmith run metadata.
"""

from __future__ import annotations

from typing import Any

from rag_app.core.interfaces import LLMProvider, LLMResponse
from rag_app.observability.provider import ObservabilityProvider


class LLMRouter(LLMProvider):
    """Routes LLM calls through primary→fallback with error handling."""

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
        obs: ObservabilityProvider,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._obs = obs

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Try primary provider; on any exception, fall back to secondary."""
        try:
            response = await self._primary.generate(prompt, **kwargs)
            self._obs.log_event(
                "llm.call.success",
                {
                    "provider": response.provider,
                    "model": response.model,
                    "tokens": response.tokens_used,
                    "latency_ms": response.latency_ms,
                },
            )
            return response

        except Exception as primary_error:
            self._obs.log_warning(
                "llm.fallback.triggered",
                {
                    "primary_provider": "groq",
                    "primary_error_type": type(primary_error).__name__,
                    "primary_error": str(primary_error)[:500],
                    "falling_back_to": "openrouter",
                },
            )

            try:
                response = await self._fallback.generate(prompt, **kwargs)
                self._obs.log_event(
                    "llm.fallback.success",
                    {
                        "provider": response.provider,
                        "model": response.model,
                        "tokens": response.tokens_used,
                        "latency_ms": response.latency_ms,
                    },
                )
                return response

            except Exception as fallback_error:
                self._obs.log_error(
                    "llm.all_providers_failed",
                    {
                        "primary_error": str(primary_error)[:500],
                        "fallback_error": str(fallback_error)[:500],
                    },
                )
                raise RuntimeError(
                    f"All LLM providers failed. "
                    f"Primary (groq): {primary_error}. "
                    f"Fallback (openrouter): {fallback_error}."
                ) from fallback_error
