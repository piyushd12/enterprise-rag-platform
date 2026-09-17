"""Asserts /chat emits a correlated LangSmith run and Logfire span.

Mocks the SDK-facing calls rather than hitting real backends:
- The compiled graph's `ainvoke` is what LangGraph/LangChain use to create
  a LangSmith run from the `config` passed to it, so asserting `ainvoke`
  was awaited with the expected `run_name` + `metadata` is the practical
  way to assert "a LangSmith run was emitted with expected metadata"
  without a real LangSmith account/network call.
- `logfire.span` is patched directly and asserted to have been called
  with the same request_id/query, proving the two tools stay correlated.

Fully mocked below the route boundary, so no real LLM/Qdrant/Redis/API
keys are needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from rag_app.api.dependencies import (
    get_cache,
    get_embedding_provider,
    get_keyword_search,
    get_llm_provider,
    get_reranker,
    get_vector_store,
)
from rag_app.main import app


def test_chat_emits_correlated_langsmith_run_and_logfire_span():
    fake_result = {
        "cache_hit": False,
        "generated_answer": "Test answer.",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "llm_provider_used": "stub",
        "llm_model_used": "stub-model",
    }
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value=fake_result)

    # Avoid constructing the real (network/API-key-dependent) singletons --
    # the graph itself is mocked out below, so these just need to exist.
    app.dependency_overrides[get_embedding_provider] = lambda: object()
    app.dependency_overrides[get_vector_store] = lambda: object()
    app.dependency_overrides[get_llm_provider] = lambda: object()
    app.dependency_overrides[get_cache] = lambda: None
    app.dependency_overrides[get_reranker] = lambda: object()
    app.dependency_overrides[get_keyword_search] = lambda: object()

    try:
        with (
            patch(
                "rag_app.api.routes.chat.build_rag_graph", return_value=fake_graph
            ) as mock_build_graph,
            patch("rag_app.observability.provider.logfire.span") as mock_logfire_span,
        ):
            mock_logfire_span.return_value.__exit__.return_value = False
            client = TestClient(app)
            response = client.post("/chat", json={"query": "What is RAG?", "top_k": 5})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    request_id = response.json()["request_id"]

    # --- LangSmith run: graph.ainvoke's `config` is what creates it ---
    mock_build_graph.assert_called_once()
    fake_graph.ainvoke.assert_awaited_once()
    _, call_kwargs = fake_graph.ainvoke.call_args
    assert call_kwargs["config"]["run_name"] == "rag_pipeline"
    assert call_kwargs["config"]["metadata"]["request_id"] == request_id
    assert call_kwargs["config"]["metadata"]["query"] == "What is RAG?"

    # --- Logfire span: same request_id, proving cross-tool correlation ---
    mock_logfire_span.assert_called_once()
    span_args, span_kwargs = mock_logfire_span.call_args
    assert span_args[0] == "chat.request"
    assert span_kwargs["request_id"] == request_id
    assert span_kwargs["query"] == "What is RAG?"
