import logging
import re
import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a string."""
    return len(_tokenizer.encode(text))


def _estimate_page_num(
    char_start: int,
    page_boundaries: list[int],
) -> int:
    if not page_boundaries:
        return 1
    for i, boundary in enumerate(reversed(page_boundaries)):
        if char_start >= boundary:
            return len(page_boundaries) - i
    return 1


def chunk_document(
    text: str,
    document_id: str,
    workspace_id: str,
    page_boundaries: list[int] | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    min_chunk_size: int = 50,
) -> list[dict]:
    """
    Returns a list of dicts, each representing one chunk:
    {
        "id": str,
        "document_id": str,
        "workspace_id": str,
        "chunk_index": int,
        "chunk_text": str,
        "page_num": int,
        "char_start": int,
        "char_end": int,
        "token_count": int,
    }
    """
    if not text or not text.strip():
        logger.warning(f"Empty text passed to chunker for document {document_id}")
        return []

    char_chunk_size = chunk_size * 4
    char_overlap = chunk_overlap * 4

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_chunk_size,
        chunk_overlap=char_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
    )

    lc_chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": document_id}],
    )

    chunks = []
    for i, lc_chunk in enumerate(lc_chunks):
        chunk_text = lc_chunk.page_content.strip()

        if len(chunk_text) < 20 or count_tokens(chunk_text) < min_chunk_size:
            logger.debug(f"Skipping tiny chunk {i} (too short)")
            continue

        clean_text = re.sub(r"\[Page \d+\]\n?", "", chunk_text).strip()
        if not clean_text:
            continue

        char_start = lc_chunk.metadata.get("start_index", 0)
        char_end = char_start + len(chunk_text)

        chunk = {
            "id": f"{document_id}_{i}",
            "document_id": document_id,
            "workspace_id": workspace_id,
            "chunk_index": i,
            "chunk_text": clean_text,
            "page_num": _estimate_page_num(char_start, page_boundaries or []),
            "char_start": char_start,
            "char_end": char_end,
            "token_count": count_tokens(clean_text),
        }
        chunks.append(chunk)

    logger.info(
        f"Chunked document {document_id}: "
        f"{len(lc_chunks)} raw → {len(chunks)} usable chunks"
    )
    return chunks