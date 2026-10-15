"""
Groq LLM provider — primary provider using free-tier models.

Uses langchain-groq under the hood, but exposes only the LLMProvider interface.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_groq import ChatGroq

from rag_app.config.settings import settings
from rag_app.core.interfaces import LLMProvider, LLMResponse


def _as_input(prompt: str, system: str | None) -> str | list[tuple[str, str]]:
    """A separate system message when one is given, else the bare prompt
    (sent as a single user message, as before)."""
    return [("system", system), ("human", prompt)] if system else prompt


class GroqProvider(LLMProvider):
    """LLMProvider implementation using Groq's free-tier API."""

    PROVIDER_NAME = "groq"

    def __init__(self, model: str | None = None) -> None:
        self._model_name = model or settings.llm_primary_model
        self._llm = ChatGroq(
            model=self._model_name,
            api_key=settings.groq_api_key,
            temperature=0,
            # 1024 was too tight -- observed truncating mid-answer on
            # table-formatted responses (a malformed final row rather
            # than a clean cutoff).
            max_tokens=2048,
        )

    async def generate(
        self, prompt: str, system: str | None = None, **kwargs: Any
    ) -> LLMResponse:
        """Generate a completion via Groq."""
        start = time.perf_counter()

        response = await self._llm.ainvoke(_as_input(prompt, system))
        latency = (time.perf_counter() - start) * 1000

        # Extract token usage from response metadata
        usage = response.usage_metadata or {}
        tokens = usage.get("total_tokens", 0)

        return LLMResponse(
            content=response.content,
            provider=self.PROVIDER_NAME,
            model=self._model_name,
            tokens_used=tokens,
            latency_ms=round(latency, 2),
        )

    async def stream(
        self, prompt: str, meta: dict[str, Any], system: str | None = None
    ) -> AsyncIterator[str]:
        """Stream a completion token-by-token via Groq."""
        start = time.perf_counter()
        tokens_used = 0

        async for chunk in self._llm.astream(_as_input(prompt, system)):
            if chunk.usage_metadata:
                tokens_used = chunk.usage_metadata.get("total_tokens", tokens_used)
            if chunk.content:
                yield chunk.content

        meta["provider"] = self.PROVIDER_NAME
        meta["model"] = self._model_name
        meta["tokens_used"] = tokens_used
        meta["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
