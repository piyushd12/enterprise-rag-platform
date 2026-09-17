"""Unit tests for LLMRouter's primary -> fallback logic (mocked LLMProvider)."""

from __future__ import annotations

import pytest

from rag_app.core.interfaces import LLMProvider, LLMResponse
from rag_app.llm.router import LLMRouter
from rag_app.observability.provider import ObservabilityProvider


class _StubProvider(LLMProvider):
    """Minimal LLMProvider that either returns a canned response or raises."""

    def __init__(self, *, response: LLMResponse | None = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls = 0

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _response(provider: str) -> LLMResponse:
    return LLMResponse(
        content=f"answer from {provider}",
        provider=provider,
        model="test-model",
        tokens_used=10,
        latency_ms=1.0,
    )


async def test_primary_success_never_calls_fallback():
    primary = _StubProvider(response=_response("groq"))
    fallback = _StubProvider(response=_response("openrouter"))
    router = LLMRouter(primary=primary, fallback=fallback, obs=ObservabilityProvider())

    result = await router.generate("hello")

    assert result.provider == "groq"
    assert primary.calls == 1
    assert fallback.calls == 0


async def test_primary_failure_triggers_fallback():
    primary = _StubProvider(error=RuntimeError("groq down"))
    fallback = _StubProvider(response=_response("openrouter"))
    router = LLMRouter(primary=primary, fallback=fallback, obs=ObservabilityProvider())

    result = await router.generate("hello")

    assert result.provider == "openrouter"
    assert primary.calls == 1
    assert fallback.calls == 1


async def test_both_providers_failing_raises_with_both_errors():
    primary = _StubProvider(error=RuntimeError("groq down"))
    fallback = _StubProvider(error=RuntimeError("openrouter down"))
    router = LLMRouter(primary=primary, fallback=fallback, obs=ObservabilityProvider())

    with pytest.raises(RuntimeError) as exc_info:
        await router.generate("hello")

    assert "groq down" in str(exc_info.value)
    assert "openrouter down" in str(exc_info.value)
