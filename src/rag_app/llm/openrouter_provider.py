"""
OpenRouter LLM provider — fallback provider using free Nemotron model.

Uses langchain-openai with OpenRouter's OpenAI-compatible endpoint.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_openai import ChatOpenAI

from rag_app.config.settings import settings
from rag_app.core.interfaces import LLMProvider, LLMResponse


class OpenRouterProvider(LLMProvider):
    """LLMProvider implementation using OpenRouter's free-tier models."""

    PROVIDER_NAME = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str | None = None) -> None:
        self._model_name = model or settings.llm_fallback_model
        self._llm = ChatOpenAI(
            model=self._model_name,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=self.BASE_URL,
            temperature=0,
            max_tokens=1024,
        )

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion via OpenRouter."""
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
