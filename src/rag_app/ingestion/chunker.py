"""
Text chunker with configurable chunk size and overlap.

Uses a recursive character-based splitting strategy that tries to
preserve paragraph/sentence boundaries before resorting to arbitrary cuts.
"""

from __future__ import annotations

from rag_app.config.settings import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks.

    Splitting priority: double-newline → single-newline → sentence → space → char.

    Args:
        text: The full document text.
        chunk_size: Max characters per chunk (default from settings).
        chunk_overlap: Overlap between consecutive chunks (default from settings).

    Returns:
        List of text chunks.
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " ", ""]
    return _recursive_split(text, separators, size, overlap)


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Recursively split text using a hierarchy of separators."""
    final_chunks: list[str] = []

    # Find the best separator that exists in the text
    separator = separators[-1]
    for sep in separators:
        if sep in text:
            separator = sep
            break

    # Split on the chosen separator
    splits = text.split(separator) if separator else list(text)

    current_chunk: list[str] = []
    current_length = 0

    for split in splits:
        piece = split.strip()
        if not piece:
            continue

        piece_len = len(piece) + (len(separator) if current_chunk else 0)

        if current_length + piece_len > chunk_size and current_chunk:
            # Emit current chunk
            chunk_text_joined = separator.join(current_chunk).strip()
            if chunk_text_joined:
                final_chunks.append(chunk_text_joined)

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

        current_chunk.append(piece)
        current_length += piece_len

    # Emit remaining
    if current_chunk:
        chunk_text_joined = separator.join(current_chunk).strip()
        if chunk_text_joined:
            final_chunks.append(chunk_text_joined)

    return final_chunks
