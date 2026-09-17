"""Unit tests for the recursive text chunker.

Includes regression coverage for a real bug found in this project: pieces
still larger than chunk_size after splitting on one separator used to be
emitted as-is instead of recursing into a finer separator, producing
chunks up to ~27x the configured size. Every chunk-size assertion here
guards against that regressing.
"""

from __future__ import annotations

from itertools import pairwise

from rag_app.ingestion.chunker import chunk_text


def test_short_text_returns_a_single_chunk():
    text = "This is a short document."
    assert chunk_text(text, chunk_size=500, chunk_overlap=50) == [text]


def test_empty_or_blank_text_returns_no_chunks():
    assert chunk_text("", chunk_size=500, chunk_overlap=50) == []
    assert chunk_text("   \n  ", chunk_size=500, chunk_overlap=50) == []


def test_every_chunk_respects_the_size_limit_even_with_no_newlines():
    # One long run-on paragraph forces the splitter all the way down to
    # sentence/word-level separators -- the exact shape that used to
    # produce grossly oversized chunks before the recursive split was fixed.
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * 100  # ~4600 chars
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_a_single_word_longer_than_chunk_size_is_hard_sliced():
    text = "x" * 1000
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)

    assert len(chunks) == 10
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_paragraph_boundaries_are_preferred_over_mid_paragraph_cuts():
    text = "First paragraph.\n\n" + ("word " * 60).strip() + "\n\nThird paragraph."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)

    assert all(len(c) <= 100 for c in chunks)
    # Short enough to stand alone rather than being fused mid-word with
    # the much longer paragraph that follows it.
    assert chunks[0] == "First paragraph."
    # The final short paragraph may still be greedily merged with
    # trailing content from the previous (much longer) paragraph if it
    # fits within chunk_size -- that's intended, so just check no content
    # was lost rather than requiring an exact standalone chunk.
    assert any("Third paragraph." in c for c in chunks)


def test_consecutive_chunks_share_overlapping_content():
    text = "".join(f"Sentence number {i}. " for i in range(50))
    chunks = chunk_text(text, chunk_size=150, chunk_overlap=30)

    assert len(chunks) > 1
    # The end of each chunk should reappear at the start of the next one.
    for first, second in pairwise(chunks):
        last_piece_of_first = first.split(". ")[-1]
        assert last_piece_of_first in second
