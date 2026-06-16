from app.services.rag.context_builder import build_context, format_history_for_llm


def _make_chunks(n: int, tokens_each: int = 100) -> list[dict]:
    """Create n fake chunks of approximately tokens_each tokens each."""
    word = "information " * (tokens_each // 2)
    return [
        {
            "chunk_text": f"Chunk {i}: {word}",
            "document_id": f"doc-{i}",
            "page_num": i + 1,
            "score": round(0.99 - i * 0.05, 2),
            "chunk_index": i,
        }
        for i in range(n)
    ]


def test_build_context_returns_string_and_list():
    chunks = _make_chunks(3, tokens_each=50)
    context, used = build_context(chunks, max_tokens=500)
    assert isinstance(context, str)
    assert isinstance(used, list)


def test_build_context_respects_token_budget():
    chunks = _make_chunks(10, tokens_each=200)
    context, used = build_context(chunks, max_tokens=300)
    assert len(used) < len(chunks)
    assert len(used) >= 1


def test_build_context_uses_best_chunks_first():
    chunks = _make_chunks(5, tokens_each=200)
    context, used = build_context(chunks, max_tokens=300)
    if used:
        assert used[0]["document_id"] == "doc-0"


def test_build_context_includes_source_labels():
    chunks = _make_chunks(2, tokens_each=50)
    context, used = build_context(chunks, max_tokens=1000)
    assert "[Source 1" in context
    assert "Page" in context


def test_build_context_empty_chunks():
    context, used = build_context([], max_tokens=3000)
    assert context == ""
    assert used == []


def test_build_context_all_chunks_fit():
    chunks = _make_chunks(3, tokens_each=50)
    context, used = build_context(chunks, max_tokens=3000)
    assert len(used) == 3


def test_format_history_returns_correct_role_strings():
    from unittest.mock import MagicMock
    from app.models.conversation import MessageRole

    msg1 = MagicMock()
    msg1.role = MessageRole.USER
    msg1.content = "What is this document about?"

    msg2 = MagicMock()
    msg2.role = MessageRole.ASSISTANT
    msg2.content = "This document is about..."

    history = format_history_for_llm([msg1, msg2])
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[0]["content"] == "What is this document about?"


def test_format_history_keeps_last_6():
    """Should return at most the last 6 messages."""
    from unittest.mock import MagicMock
    from app.models.conversation import MessageRole

    messages = []
    for i in range(10):
        msg = MagicMock()
        msg.role = MessageRole.USER
        msg.content = f"Message {i}"
        messages.append(msg)

    history = format_history_for_llm(messages)
    assert len(history) == 6
    assert history[0]["content"] == "Message 4"
    assert history[-1]["content"] == "Message 9"