# tests/test_reranker.py
"""
Unit tests for the cross-encoder RerankerService.

The actual model is mocked — we test the service's logic
(input/output shape, sorting, fallback, edge cases) without
downloading any model weights during the test run.
"""
import pytest
from unittest.mock import MagicMock, patch

from app.services.retrieval.reranker import RerankerService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(chunk_id: str, score: float = 0.8) -> dict:
    return {
        "chunk_id": chunk_id,
        "chunk_text": f"Content for chunk {chunk_id}. " * 5,
        "document_id": "doc-1",
        "workspace_id": "ws-test",
        "page_num": 1,
        "chunk_index": 0,
        "score": score,
    }


def make_reranker_with_mock_model(scores: list[float]) -> RerankerService:
    """Return a RerankerService with the model pre-loaded as a mock."""
    svc = RerankerService()
    mock_model = MagicMock()
    mock_model.predict.return_value = scores
    svc._model = mock_model
    return svc


# ---------------------------------------------------------------------------
# rerank() — output shape and ordering
# ---------------------------------------------------------------------------

def test_rerank_returns_top_k():
    chunks = [make_chunk(f"c{i}") for i in range(10)]
    svc = make_reranker_with_mock_model([float(i) for i in range(10)])
    result = svc.rerank("query", chunks, top_k=3)
    assert len(result) == 3


def test_rerank_orders_by_score_descending():
    chunks = [make_chunk("low"), make_chunk("mid"), make_chunk("high")]
    # Cross-encoder scores: low=1.0, mid=5.0, high=9.0
    svc = make_reranker_with_mock_model([1.0, 5.0, 9.0])
    result = svc.rerank("query", chunks, top_k=3)
    assert result[0]["chunk_id"] == "high"
    assert result[1]["chunk_id"] == "mid"
    assert result[2]["chunk_id"] == "low"


def test_rerank_can_reverse_hybrid_order():
    """The reranker should be able to promote a low-ranked hybrid chunk."""
    # Chunks arrive in hybrid order: A, B, C (A was best by RRF)
    chunks = [make_chunk("A", score=0.9), make_chunk("B", score=0.5), make_chunk("C", score=0.3)]
    # But cross-encoder says C is actually most relevant
    svc = make_reranker_with_mock_model([0.1, 0.5, 9.9])
    result = svc.rerank("query", chunks, top_k=2)
    assert result[0]["chunk_id"] == "C"
    assert result[1]["chunk_id"] == "B"


def test_rerank_attaches_rerank_score():
    chunks = [make_chunk("A"), make_chunk("B")]
    svc = make_reranker_with_mock_model([3.5, 1.2])
    result = svc.rerank("query", chunks, top_k=2)
    assert "rerank_score" in result[0]
    assert isinstance(result[0]["rerank_score"], float)


def test_rerank_score_matches_model_output():
    chunks = [make_chunk("A"), make_chunk("B")]
    svc = make_reranker_with_mock_model([7.77, 2.22])
    result = svc.rerank("query", chunks, top_k=2)
    scores_by_id = {r["chunk_id"]: r["rerank_score"] for r in result}
    assert abs(scores_by_id["A"] - 7.77) < 0.01
    assert abs(scores_by_id["B"] - 2.22) < 0.01


def test_rerank_does_not_mutate_original_chunks():
    chunks = [make_chunk("A"), make_chunk("B")]
    original_keys = set(chunks[0].keys())
    svc = make_reranker_with_mock_model([5.0, 3.0])
    svc.rerank("query", chunks, top_k=2)
    # Original chunk dicts should not have rerank_score added
    assert set(chunks[0].keys()) == original_keys


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_rerank_empty_chunks_returns_empty():
    svc = RerankerService()
    result = svc.rerank("query", [], top_k=5)
    assert result == []


def test_rerank_fewer_chunks_than_top_k_returns_all():
    """When we have fewer chunks than top_k, reranker still runs and returns all."""
    chunks = [make_chunk("A"), make_chunk("B")]
    svc = make_reranker_with_mock_model([3.0, 1.0])
    result = svc.rerank("query", chunks, top_k=10)
    # All chunks returned (can't return more than we have)
    assert len(result) == 2
    # Still sorted by rerank score
    assert result[0]["chunk_id"] == "A"


def test_rerank_top_k_one_returns_single_best():
    chunks = [make_chunk("low", 0.1), make_chunk("high", 0.9)]
    svc = make_reranker_with_mock_model([1.0, 9.0])
    result = svc.rerank("query", chunks, top_k=1)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "high"


def test_rerank_calls_model_with_correct_pairs():
    """Verify that (query, chunk_text) pairs are passed to model.predict."""
    chunks = [make_chunk("A"), make_chunk("B")]
    svc = make_reranker_with_mock_model([1.0, 2.0])
    svc.rerank("my question", chunks, top_k=2)

    call_args = svc._model.predict.call_args[0][0]
    assert call_args[0] == ("my question", chunks[0]["chunk_text"])
    assert call_args[1] == ("my question", chunks[1]["chunk_text"])


# ---------------------------------------------------------------------------
# Fallback when model predict fails
# ---------------------------------------------------------------------------

def test_rerank_falls_back_on_model_error():
    """If model.predict raises, should return top_k from hybrid order."""
    chunks = [make_chunk(f"c{i}") for i in range(10)]
    svc = RerankerService()
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("GPU OOM")
    svc._model = mock_model

    result = svc.rerank("query", chunks, top_k=3)
    # Falls back to first top_k from hybrid results
    assert len(result) == 3
    assert result[0]["chunk_id"] == "c0"


# ---------------------------------------------------------------------------
# is_loaded property
# ---------------------------------------------------------------------------

def test_is_loaded_false_when_model_not_initialized():
    svc = RerankerService()
    assert svc.is_loaded is False


def test_is_loaded_true_when_model_set():
    svc = RerankerService()
    svc._model = MagicMock()
    assert svc.is_loaded is True
