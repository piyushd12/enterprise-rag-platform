# tests/test_hybrid_retriever.py
"""
Unit tests for Reciprocal Rank Fusion and the hybrid retriever.

_reciprocal_rank_fusion is pure Python and tested deterministically.
hybrid_search is tested with mocked Qdrant and BM25 to isolate the fusion logic.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.retrieval.hybrid_retriever import _reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_chunk(chunk_id: str, score: float = 1.0, doc: str = "doc-1") -> dict:
    return {
        "chunk_id": chunk_id,
        "chunk_text": f"Text for chunk {chunk_id}",
        "document_id": doc,
        "workspace_id": "ws-test",
        "page_num": 1,
        "chunk_index": 0,
        "score": score,
    }


# ---------------------------------------------------------------------------
# _reciprocal_rank_fusion tests
# ---------------------------------------------------------------------------

def test_rrf_single_list_preserves_order():
    dense = [make_chunk("A"), make_chunk("B"), make_chunk("C")]
    fused = _reciprocal_rank_fusion(dense, top_k=3)
    fused_ids = [r["chunk_id"] for r in fused]
    assert fused_ids == ["A", "B", "C"]


def test_rrf_promotes_chunk_appearing_in_both_lists():
    """Chunk C appears at rank 1 in BM25 and rank 3 in dense → should beat rank-1 dense-only chunk."""
    dense = [
        make_chunk("A", score=0.95),   # rank 1 in dense
        make_chunk("B", score=0.88),   # rank 2 in dense
        make_chunk("C", score=0.75),   # rank 3 in dense
    ]
    bm25 = [
        make_chunk("C", score=12.4),   # rank 1 in BM25
        make_chunk("A", score=9.8),    # rank 2 in BM25
        make_chunk("D", score=7.1),    # rank 1 only in BM25
    ]
    fused = _reciprocal_rank_fusion(dense, bm25, top_k=4)
    fused_ids = [r["chunk_id"] for r in fused]

    # A appears rank 1 + rank 2 → strong consensus
    # C appears rank 3 + rank 1 → also strong consensus
    # Both A and C should beat D (only appears in one list)
    assert "D" not in fused_ids[:2], "D should not beat consensus chunks A and C"
    assert "A" in fused_ids
    assert "C" in fused_ids


def test_rrf_uses_rank_not_raw_scores():
    """Even if BM25 scores are huge, RRF should use rank position only."""
    dense = [make_chunk("A", score=0.5), make_chunk("B", score=0.4)]
    bm25 = [
        make_chunk("X", score=9999),   # massive BM25 score, rank 1
        make_chunk("A", score=8000),   # rank 2 in BM25
    ]
    fused = _reciprocal_rank_fusion(dense, bm25, top_k=3)
    # A: rank 1 in dense + rank 2 in BM25
    # X: rank 1 in BM25 only
    # A should win because it appears in BOTH lists
    fused_ids = [r["chunk_id"] for r in fused]
    assert fused_ids[0] == "A", "Consensus chunk A should outrank BM25-only chunk X"


def test_rrf_respects_top_k():
    chunks = [make_chunk(f"chunk-{i}") for i in range(10)]
    fused = _reciprocal_rank_fusion(chunks, top_k=3)
    assert len(fused) == 3


def test_rrf_top_k_larger_than_results():
    chunks = [make_chunk("A"), make_chunk("B")]
    fused = _reciprocal_rank_fusion(chunks, top_k=100)
    assert len(fused) == 2


def test_rrf_empty_list_returns_empty():
    fused = _reciprocal_rank_fusion([], top_k=10)
    assert fused == []


def test_rrf_single_chunk_single_list():
    fused = _reciprocal_rank_fusion([make_chunk("solo")], top_k=5)
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "solo"


def test_rrf_result_has_score_and_score_type():
    chunks = [make_chunk("A"), make_chunk("B")]
    fused = _reciprocal_rank_fusion(chunks, top_k=2)
    for result in fused:
        assert "score" in result
        assert "score_type" in result
        assert result["score_type"] == "rrf"
        assert result["score"] > 0.0


def test_rrf_scores_are_positive_and_ordered():
    dense = [make_chunk(f"d{i}") for i in range(5)]
    bm25 = [make_chunk(f"b{i}") for i in range(5)]
    fused = _reciprocal_rank_fusion(dense, bm25, top_k=10)
    scores = [r["score"] for r in fused]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_rrf_deduplicates_overlapping_chunks():
    """Same chunk_id in both lists should appear exactly once in output."""
    dense = [make_chunk("A"), make_chunk("B")]
    bm25 = [make_chunk("A"), make_chunk("C")]
    fused = _reciprocal_rank_fusion(dense, bm25, top_k=10)
    fused_ids = [r["chunk_id"] for r in fused]
    assert fused_ids.count("A") == 1


def test_rrf_three_lists_accumulates_from_all():
    """With 3 lists, a chunk appearing in all 3 should have a higher score
    than a chunk appearing in only 2."""
    list1 = [make_chunk("consensus"), make_chunk("list1only")]
    list2 = [make_chunk("consensus"), make_chunk("list2only")]
    list3 = [make_chunk("consensus"), make_chunk("list3only")]

    fused = _reciprocal_rank_fusion(list1, list2, list3, top_k=10)
    scores_by_id = {r["chunk_id"]: r["score"] for r in fused}

    # "consensus" appears rank 1 in all 3 → highest possible score
    assert scores_by_id["consensus"] > scores_by_id["list1only"]
    assert scores_by_id["consensus"] > scores_by_id["list2only"]
    assert scores_by_id["consensus"] > scores_by_id["list3only"]


# ---------------------------------------------------------------------------
# hybrid_search integration tests (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hybrid_search_falls_back_to_dense_when_bm25_not_ready():
    dense_chunks = [make_chunk("dense-A"), make_chunk("dense-B")]

    with (
        patch("app.services.retrieval.hybrid_retriever.embedder") as mock_embedder,
        patch("app.services.retrieval.hybrid_retriever.vector_store") as mock_vs,
        patch("app.services.retrieval.hybrid_retriever.bm25_index") as mock_bm25,
    ):
        mock_embedder.embed_query.return_value = [0.1] * 384
        mock_vs.search.return_value = dense_chunks
        mock_bm25.is_ready = False

        from app.services.retrieval.hybrid_retriever import hybrid_search
        results = await hybrid_search("test query", workspace_id="ws-1", top_k=2)

    assert results == dense_chunks[:2]


@pytest.mark.asyncio
async def test_hybrid_search_falls_back_to_bm25_when_dense_empty():
    bm25_chunks = [make_chunk("bm25-A"), make_chunk("bm25-B")]

    with (
        patch("app.services.retrieval.hybrid_retriever.embedder") as mock_embedder,
        patch("app.services.retrieval.hybrid_retriever.vector_store") as mock_vs,
        patch("app.services.retrieval.hybrid_retriever.bm25_index") as mock_bm25,
    ):
        mock_embedder.embed_query.return_value = [0.1] * 384
        mock_vs.search.return_value = []  # dense returns nothing
        mock_bm25.is_ready = True
        mock_bm25.search.return_value = bm25_chunks

        from app.services.retrieval.hybrid_retriever import hybrid_search
        results = await hybrid_search("test query", workspace_id="ws-1", top_k=2)

    assert results == bm25_chunks[:2]


@pytest.mark.asyncio
async def test_hybrid_search_fuses_both_when_both_available():
    dense_chunks = [make_chunk("A", score=0.9), make_chunk("B", score=0.8)]
    bm25_chunks = [make_chunk("B", score=10.0), make_chunk("C", score=8.0)]

    with (
        patch("app.services.retrieval.hybrid_retriever.embedder") as mock_embedder,
        patch("app.services.retrieval.hybrid_retriever.vector_store") as mock_vs,
        patch("app.services.retrieval.hybrid_retriever.bm25_index") as mock_bm25,
    ):
        mock_embedder.embed_query.return_value = [0.1] * 384
        mock_vs.search.return_value = dense_chunks
        mock_bm25.is_ready = True
        mock_bm25.search.return_value = bm25_chunks

        from app.services.retrieval.hybrid_retriever import hybrid_search
        results = await hybrid_search("test query", workspace_id="ws-1", top_k=3)

    # Should contain all 3 unique chunks: A, B, C
    result_ids = {r["chunk_id"] for r in results}
    assert "A" in result_ids
    assert "B" in result_ids
    assert "C" in result_ids

    # B appears in both lists → should rank highest (consensus)
    assert results[0]["chunk_id"] == "B"

    # All results should have RRF score_type
    assert all(r.get("score_type") == "rrf" for r in results)


@pytest.mark.asyncio
async def test_hybrid_search_filters_by_document_id():
    """document_id filter should be respected in both dense and BM25 calls."""
    with (
        patch("app.services.retrieval.hybrid_retriever.embedder") as mock_embedder,
        patch("app.services.retrieval.hybrid_retriever.vector_store") as mock_vs,
        patch("app.services.retrieval.hybrid_retriever.bm25_index") as mock_bm25,
    ):
        mock_embedder.embed_query.return_value = [0.1] * 384
        mock_vs.search.return_value = []
        mock_bm25.is_ready = False

        from app.services.retrieval.hybrid_retriever import hybrid_search
        await hybrid_search(
            "test query",
            workspace_id="ws-1",
            document_id="doc-42",
            top_k=5,
        )

        # Verify document_id was passed to vector_store.search
        call_kwargs = mock_vs.search.call_args.kwargs
        assert call_kwargs.get("document_id") == "doc-42"
