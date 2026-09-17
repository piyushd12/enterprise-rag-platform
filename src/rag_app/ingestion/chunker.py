"""
Text chunker with configurable chunk size and overlap.

Uses a recursive character-based splitting strategy that tries to
preserve paragraph/sentence boundaries before resorting to arbitrary cuts.
"""

from __future__ import annotations

from rag_app.config.settings import settings

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks, each at most chunk_size characters.

    Splitting priority: double-newline -> single-newline -> sentence -> space -> char.
    Every returned chunk is guaranteed to be <= chunk_size characters: a piece
    that is still too large after splitting on one separator is recursively
    split again on the next, finer separator (falling back to a hard
    character slice only once no separator remains).

    Args:
        text: The full document text.
        chunk_size: Max characters per chunk (default from settings).
        chunk_overlap: Overlap between consecutive chunks (default from settings).

    Returns:
        List of text chunks.
    """
    # `or` would treat an explicit 0 (e.g. "no overlap") as falsy and
    # silently fall back to the configured default -- use an explicit
    # None-check so callers can actually request 0.
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    return _split_recursive(text, _SEPARATORS, size, overlap)


def _split_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text on the best available separator, recursing into any
    resulting piece that is still larger than chunk_size."""
    separator = separators[-1]
    remaining_separators: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_separators = separators[i + 1 :]
            break

    splits = text.split(separator) if separator else list(text)

    good_splits: list[str] = []
    for split in splits:
        piece = split.strip() if separator else split
        if not piece:
            continue
        if len(piece) <= chunk_size:
            good_splits.append(piece)
        elif remaining_separators:
            good_splits.extend(
                _split_recursive(piece, remaining_separators, chunk_size, chunk_overlap)
            )
        else:
            # No finer separator left -- hard character slice as a last resort.
            good_splits.extend(
                piece[i : i + chunk_size] for i in range(0, len(piece), chunk_size)
            )

    return _merge_splits(good_splits, separator, chunk_size, chunk_overlap)


def _merge_splits(
    splits: list[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Greedily merge small pieces into chunks up to chunk_size, carrying
    trailing overlap forward into the next chunk."""
    final_chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for piece in splits:
        piece_len = len(piece) + (len(separator) if current_chunk else 0)

        if current_length + piece_len > chunk_size and current_chunk:
            chunk_joined = separator.join(current_chunk).strip()
            if chunk_joined:
                final_chunks.append(chunk_joined)

            # Keep overlap from the end of the current chunk
            overlap_chunks: list[str] = []
            overlap_len = 0
            for prev_piece in reversed(current_chunk):
                if overlap_len + len(prev_piece) > chunk_overlap:
                    break
                overlap_chunks.insert(0, prev_piece)
                overlap_len += len(prev_piece) + len(separator)

            current_chunk = overlap_chunks
            current_length = sum(len(c) for c in current_chunk) + len(separator) * max(
                0, len(current_chunk) - 1
            )
            piece_len = len(piece) + (len(separator) if current_chunk else 0)

        current_chunk.append(piece)
        current_length += piece_len

    if current_chunk:
        chunk_joined = separator.join(current_chunk).strip()
        if chunk_joined:
            final_chunks.append(chunk_joined)

    return final_chunks
