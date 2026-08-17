"""
Groq LLM provider — primary provider using free-tier models.

Uses langchain-groq under the hood, but exposes only the LLMProvider interface.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_groq import ChatGroq

from rag_app.config.settings import settings
from rag_app.core.interfaces import LLMProvider, LLMResponse


class GroqProvider(LLMProvider):
    """LLMProvider implementation using Groq's free-tier API."""

    PROVIDER_NAME = "groq"

    def __init__(self, model: str | None = None) -> None:
        self._model_name = model or settings.llm_primary_model
        self._llm = ChatGroq(
            model=self._model_name,
            api_key=settings.groq_api_key,
            temperature=0,
            max_tokens=1024,
        )

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion via Groq."""
        start = time.perf_counter()

        response = await self._llm.ainvoke(prompt)
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
