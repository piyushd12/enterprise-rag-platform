from app.services.chunker import chunk_document, count_tokens


def _make_long_text(paragraphs: int = 10) -> str:
    paragraph = (
        "Artificial intelligence is rapidly changing the way we work and live. "
        "Machine learning models can now perform tasks that were once thought to "
        "be exclusively human capabilities. Natural language processing has made "
        "enormous strides in recent years. "
    )
    return "\n\n".join([paragraph] * paragraphs)


def test_chunker_produces_chunks():
    text = _make_long_text(10)
    chunks = chunk_document(
        text=text,
        document_id="doc-001",
        workspace_id="ws-001",
    )
    assert len(chunks) > 0


def test_chunker_each_chunk_has_required_fields():
    text = _make_long_text(5)
    chunks = chunk_document(text=text, document_id="doc-002", workspace_id="ws-002")
    for chunk in chunks:
        assert "id" in chunk
        assert "document_id" in chunk
        assert "workspace_id" in chunk
        assert "chunk_text" in chunk
        assert "chunk_index" in chunk
        assert "page_num" in chunk
        assert "token_count" in chunk
        assert chunk["document_id"] == "doc-002"
        assert chunk["workspace_id"] == "ws-002"


def test_chunker_ids_are_unique():
    text = _make_long_text(20)
    chunks = chunk_document(text=text, document_id="doc-003", workspace_id="ws-003")
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))

def test_chunker_indices_are_sequential():
    text = _make_long_text(20)
    chunks = chunk_document(text=text, document_id="doc-004", workspace_id="ws-004")
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunker_respects_min_chunk_size():
    chunks = chunk_document(
        text="Hi.",
        document_id="doc-005",
        workspace_id="ws-005",
        min_chunk_size=50,
    )
    assert len(chunks) == 0


def test_chunker_token_count_is_accurate():
    text = _make_long_text(5)
    chunks = chunk_document(text=text, document_id="doc-006", workspace_id="ws-006")
    for chunk in chunks:
        actual_tokens = count_tokens(chunk["chunk_text"])
        assert abs(chunk["token_count"] - actual_tokens) <= 5


def test_chunker_returns_empty_for_blank_text():
    chunks = chunk_document(
        text="   \n\n   ",
        document_id="doc-007",
        workspace_id="ws-007",
    )
    assert chunks == []


def test_chunker_strips_page_labels():
    text = "[Page 1]\nThis is the content.\n\n[Page 2]\nMore content here."
    chunks = chunk_document(text=text, document_id="doc-008", workspace_id="ws-008")
    for chunk in chunks:
        assert "[Page" not in chunk["chunk_text"]