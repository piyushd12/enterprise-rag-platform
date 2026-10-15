"""Route tests for guardrails on both chat paths.

/chat runs the LangGraph pipeline and /chat/stream calls the node
functions directly, so every guard is tested on each path separately --
a check that only exists in one of them would otherwise pass unnoticed.

Fully in-process (dependency_overrides on every provider), so these never
skip for missing Qdrant/Redis/API keys.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_app.api.dependencies import (
    get_cache,
    get_chat_store,
    get_embedding_provider,
    get_keyword_search,
    get_llm_provider,
    get_reranker,
    get_vector_store,
)
from rag_app.api.rate_limit import limiter
from rag_app.main import app

ENDPOINTS = ["/chat", "/chat/stream"]


@pytest.fixture
def client():
    # Many requests from one test client would trip the 20/minute limit.
    limiter.enabled = False
    for dep in (
        get_embedding_provider,
        get_vector_store,
        get_llm_provider,
        get_cache,
        get_reranker,
        get_keyword_search,
        get_chat_store,
    ):
        app.dependency_overrides[dep] = lambda: object()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        limiter.enabled = True


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "filters",
    [
        {"filename": "x.pdf"},  # key not in the allowlist
        {"source_id": ["a", "b"]},  # non-scalar value
        {"source_id": {"$ne": "x"}},  # operator-style object
    ],
)
def test_disallowed_filters_get_422(client, endpoint, filters):
    response = client.post(endpoint, json={"query": "What is RAG?", "filters": filters})
    assert response.status_code == 422
