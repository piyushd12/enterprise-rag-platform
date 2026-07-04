# tests/test_bm25_index.py
"""
Unit tests for the BM25 in-memory index.

These tests run entirely in memory — no database, no Qdrant, no HTTP calls.
They verify: build, search, workspace isolation, add_chunks, remove_document.
"""
import pytest
from app.services.retrieval.bm25_index import BM25Index, _tokenize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WORKSPACE_A = "ws-aaaaaaaaaa"
WORKSPACE_B = "ws-bbbbbbbbbb"

SAMPLE_CHUNKS_A = [
    {
        "id": "a1",
        "workspace_id": WORKSPACE_A,
        "document_id": "doc-1",
        "chunk_text": "Piyush Sudhir Deshmukh is a research candidate at SIES Graduate School.",
        "page_num": 1,
        "chunk_index": 0,
    },
    {
        "id": "a2",
        "workspace_id": WORKSPACE_A,
        "document_id": "doc-1",
        "chunk_text": "The strategy design pattern uses an interface for interchangeable algorithms.",
        "page_num": 2,
        "chunk_index": 1,
    },
    {
        "id": "a3",
        "workspace_id": WORKSPACE_A,
        "document_id": "doc-2",
        "chunk_text": "Artificial intelligence has grown exponentially in recent years.",
        "page_num": 1,
        "chunk_index": 0,
    },
]

SAMPLE_CHUNKS_B = [
    {
        "id": "b1",
        "workspace_id": WORKSPACE_B,
        "document_id": "doc-3",
        "chunk_text": "Completely separate workspace content about machine learning.",
        "page_num": 1,
        "chunk_index": 0,
    },
]


@pytest.fixture
def built_index():
    """BM25Index pre-built with SAMPLE_CHUNKS_A and SAMPLE_CHUNKS_B."""
    index = BM25Index()
    index.build(SAMPLE_CHUNKS_A + SAMPLE_CHUNKS_B)
    return index


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

def test_tokenize_lowercases():
    tokens = _tokenize("Hello World")
    assert all(t == t.lower() for t in tokens)


def test_tokenize_strips_punctuation():
    tokens = _tokenize("hello, world! foo.")
    assert "," not in tokens
    assert "!" not in tokens
    assert "." not in tokens


def test_tokenize_removes_single_chars():
    tokens = _tokenize("a b c hello")
    assert "a" not in tokens
    assert "b" not in tokens
    assert "hello" in tokens


def test_tokenize_empty_string():
    assert _tokenize("") == []


def test_tokenize_punctuation_only():
    assert _tokenize("!!! ???") == []


# ---------------------------------------------------------------------------
# Build tests
# ---------------------------------------------------------------------------

def test_build_sets_total_chunks(built_index):
    assert built_index.total_chunks == len(SAMPLE_CHUNKS_A) + len(SAMPLE_CHUNKS_B)


def test_build_is_ready(built_index):
    assert built_index.is_ready is True


def test_empty_index_not_ready():
    index = BM25Index()
    assert index.is_ready is False


def test_build_with_empty_list_leaves_index_not_built():
    index = BM25Index()
    index.build([])
    assert index.is_ready is False


def test_rebuild_replaces_old_index():
    index = BM25Index()
    index.build(SAMPLE_CHUNKS_A)
    index.build(SAMPLE_CHUNKS_B)   # rebuild
    assert index.total_chunks == len(SAMPLE_CHUNKS_B)


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------

def test_search_returns_top_result_for_exact_name(built_index):
    results = built_index.search("Piyush Sudhir Deshmukh", workspace_id=WORKSPACE_A)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "a1"


def test_search_returns_results_in_score_order(built_index):
    results = built_index.search("design pattern", workspace_id=WORKSPACE_A)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_top_k(built_index):
    results = built_index.search("the", workspace_id=WORKSPACE_A, top_k=1)
    assert len(results) <= 1


def test_search_result_has_required_fields(built_index):
    results = built_index.search("research", workspace_id=WORKSPACE_A)
    assert len(results) > 0
    result = results[0]
    assert "chunk_id" in result
    assert "chunk_text" in result
    assert "document_id" in result
    assert "workspace_id" in result
    assert "page_num" in result
    assert "chunk_index" in result
    assert "score" in result


def test_search_empty_query_returns_empty(built_index):
    results = built_index.search("", workspace_id=WORKSPACE_A)
    assert results == []


def test_search_on_empty_index_returns_empty():
    index = BM25Index()
    results = index.search("hello", workspace_id=WORKSPACE_A)
    assert results == []


# ---------------------------------------------------------------------------
# Workspace isolation tests
# ---------------------------------------------------------------------------

def test_workspace_a_cannot_see_workspace_b_chunks(built_index):
    # Query matches chunk b1 semantically, but should not appear for WORKSPACE_A
    results = built_index.search("machine learning", workspace_id=WORKSPACE_A)
    chunk_ids = [r["chunk_id"] for r in results]
    assert "b1" not in chunk_ids


def test_workspace_b_cannot_see_workspace_a_chunks(built_index):
    results = built_index.search("Piyush Deshmukh", workspace_id=WORKSPACE_B)
    chunk_ids = [r["chunk_id"] for r in results]
    assert "a1" not in chunk_ids


def test_unknown_workspace_returns_empty(built_index):
    results = built_index.search("research", workspace_id="unknown-ws")
    assert results == []


# ---------------------------------------------------------------------------
# add_chunks / remove_document tests
# ---------------------------------------------------------------------------

def test_add_chunks_increases_total(built_index):
    before = built_index.total_chunks
    new_chunk = {
        "id": "a4",
        "workspace_id": WORKSPACE_A,
        "document_id": "doc-2",
        "chunk_text": "New chunk about neural networks and deep learning.",
        "page_num": 3,
        "chunk_index": 2,
    }
    built_index.add_chunks([new_chunk])
    assert built_index.total_chunks == before + 1


def test_add_chunks_makes_new_content_searchable(built_index):
    new_chunk = {
        "id": "a5",
        "workspace_id": WORKSPACE_A,
        "document_id": "doc-2",
        "chunk_text": "Quantum computing will revolutionize cryptography uniqueterm9z.",
        "page_num": 4,
        "chunk_index": 3,
    }
    built_index.add_chunks([new_chunk])
    results = built_index.search("uniqueterm9z", workspace_id=WORKSPACE_A)
    chunk_ids = [r["chunk_id"] for r in results]
    assert "a5" in chunk_ids


def test_remove_document_removes_its_chunks(built_index):
    # doc-1 has chunks a1 and a2
    built_index.remove_document("doc-1")
    results_a1 = built_index.search("Piyush Deshmukh", workspace_id=WORKSPACE_A)
    results_a2 = built_index.search("strategy design pattern", workspace_id=WORKSPACE_A)
    assert all(r["chunk_id"] != "a1" for r in results_a1)
    assert all(r["chunk_id"] != "a2" for r in results_a2)


def test_remove_document_keeps_other_chunks(built_index):
    """After removing doc-1 (chunks a1, a2), chunk a3 from doc-2 must still be in the index."""
    built_index.remove_document("doc-1")
    # Verify a3 remains in the index metadata
    assert "a3" in built_index._chunk_ids, "a3 should still be in the BM25 index after removing doc-1"
    # Verify a1 and a2 are gone
    assert "a1" not in built_index._chunk_ids
    assert "a2" not in built_index._chunk_ids
    # Verify workspace still has entries
    assert WORKSPACE_A in built_index._workspace_index
