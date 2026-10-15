"""Unit test: the retrieve node applies payload filters to BM25 results too."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from rag_app.core.interfaces import RetrievedChunk
from rag_app.rag.nodes import make_retrieve_node


def _chunk(source_id: str, idx: int) -> RetrievedChunk:
    return RetrievedChunk(
        content=f"{source_id}-{idx}", score=1.0, metadata={"source_id": source_id, "chunk_index": idx}
    )


async def test_bm25_results_respect_filters():
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2]
    store = MagicMock()
    store.search = AsyncMock(return_value=[_chunk("doc-a", 0)])
    keyword_search = MagicMock()
    keyword_search.search.return_value = [_chunk("doc-a", 1), _chunk("doc-b", 0)]

    retrieve = make_retrieve_node(embedder, store, use_reranker=True, keyword_search=keyword_search)
    out = await retrieve({"query": "q", "top_k": 5, "filters": {"source_id": "doc-a"}})

    ids = [(c.metadata["source_id"], c.metadata["chunk_index"]) for c in out["retrieved_chunks"]]
    assert ids == [("doc-a", 0), ("doc-a", 1)]
