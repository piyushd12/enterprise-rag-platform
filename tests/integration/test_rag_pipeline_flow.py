"""Integration test for the full ingest -> retrieve -> generate flow.

Uses the real FastEmbed embedder and a real Qdrant collection (both
already required to run this app locally), but a stubbed LLMProvider so
the test doesn't spend real Groq/OpenRouter quota. Requires Qdrant to be
reachable at settings.qdrant_url (`make infra`); skips itself otherwise.

Runs against a disposable collection, never the app's real "documents"
collection, and tears it down afterward.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_app.config.settings import settings
from rag_app.core.interfaces import LLMProvider, LLMResponse
from rag_app.embeddings.fastembed_provider import FastEmbedProvider
from rag_app.ingestion.pipeline import ingest_file
from rag_app.observability.provider import ObservabilityProvider
from rag_app.rag.graph import build_rag_graph
from rag_app.vectorstore.qdrant_store import QdrantStore

TEST_COLLECTION = f"test_rag_pipeline_flow_{uuid.uuid4().hex[:8]}"


def _qdrant_reachable() -> bool:
    try:
        QdrantClient(url=settings.qdrant_url).get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_reachable(), reason="Qdrant is not reachable at settings.qdrant_url"
)


class _StubLLM(LLMProvider):
    """Returns a canned response but records the prompt it was given, so
    the test can confirm generation actually used the retrieved context."""

    def __init__(self):
        self.last_prompt: str | None = None

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        self.last_prompt = prompt
        return LLMResponse(
            content="The mitochondria is the powerhouse of the cell.",
            provider="stub",
            model="stub-model",
            tokens_used=5,
            latency_ms=1.0,
        )


@pytest.fixture
def obs() -> ObservabilityProvider:
    return ObservabilityProvider()


@pytest.fixture
def vector_store(obs):
    embedder = FastEmbedProvider()
    store = QdrantStore(obs=obs, embedding_dimension=embedder.dimension(), collection_name=TEST_COLLECTION)
    yield store
    # Teardown: drop the disposable test collection either way.
    try:
        QdrantClient(url=settings.qdrant_url).delete_collection(TEST_COLLECTION)
    except UnexpectedResponse:
        pass


async def test_ingest_then_retrieve_then_generate(tmp_path, obs, vector_store):
    await vector_store.ensure_collection()
    embedder = FastEmbedProvider()

    doc_path = tmp_path / "biology_fact.txt"
    doc_path.write_text(
        "The mitochondria is the powerhouse of the cell. "
        "It generates most of the cell's supply of ATP."
    )

    ingest_result = await ingest_file(
        file_path=doc_path,
        embedding_provider=embedder,
        vector_store=vector_store,
        obs=obs,
        source_id="test-biology-fact",
    )
    assert ingest_result["chunk_count"] >= 1

    stub_llm = _StubLLM()
    graph = build_rag_graph(embedder, vector_store, stub_llm, cache=None)

    result = await graph.ainvoke(
        {"query": "What is the powerhouse of the cell?", "top_k": 3, "filters": None}
    )

    retrieved = result["retrieved_chunks"]
    assert len(retrieved) >= 1
    assert any("mitochondria" in c.content.lower() for c in retrieved)

    # The generate node must have built its prompt from the retrieved
    # chunk, not just the raw question.
    assert stub_llm.last_prompt is not None
    assert "mitochondria" in stub_llm.last_prompt.lower()
    assert result["generated_answer"] == "The mitochondria is the powerhouse of the cell."
