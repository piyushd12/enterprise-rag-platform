"""
Ingestion pipeline: load → chunk → embed → upsert to vector store.

Orchestrates the full document ingestion workflow with Logfire
instrumentation on each major step.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from rag_app.core.interfaces import EmbeddingProvider, VectorStore
from rag_app.ingestion.chunker import chunk_text
from rag_app.ingestion.loader import load_document
from rag_app.observability.provider import ObservabilityProvider


async def ingest_file(
    file_path: str | Path,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    obs: ObservabilityProvider,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    filename: str | None = None,
    source_id: str | None = None,
) -> dict:
    """Run the full ingestion pipeline for a single file.

    Steps:
        1. Load document (PDF/TXT/MD)
        2. Chunk text with configurable size/overlap
        3. Embed all chunks
        4. Upsert to vector store with metadata

    Args:
        filename: Original filename to record, if `file_path` is a staged
            temp file with a different name (see `load_document`).
        source_id: Caller-supplied stable identifier (e.g. content hash), if
            `file_path` isn't a stable on-disk location (see `load_document`).

    Returns:
        Summary dict with source_id, chunk_count, filename.
    """
    with obs.span("ingestion.pipeline", {"file_path": str(file_path)}) as span_data:
        # 1. Load
        docs = load_document(file_path, filename=filename, source_id=source_id)
        doc = docs[0]  # single-document loaders return a list of one
        content = doc["content"]
        metadata = doc["metadata"]
        source_id = metadata["source_id"]

        span_data["source_id"] = source_id
        span_data["filename"] = metadata.get("filename", "")
        span_data["doc_type"] = metadata.get("doc_type", "")

        # 2. Chunk
        with obs.span("ingestion.chunk", {"source_id": source_id}) as chunk_span:
            chunks = chunk_text(content, chunk_size, chunk_overlap)
            chunk_span["chunk_count"] = len(chunks)

        # 3. Embed
        with obs.span(
            "ingestion.embed",
            {"source_id": source_id, "chunk_count": len(chunks)},
        ) as embed_span:
            embeddings = embedding_provider.embed_documents(chunks)
            embed_span["embedding_dimension"] = (
                len(embeddings[0]) if embeddings else 0
            )

        # 4. Prepare and upsert
        ingested_at = datetime.now(timezone.utc).isoformat()
        chunk_dicts = [
            {
                "id": str(uuid.uuid4()),
                "embedding": emb,
                "content": text,
                "metadata": {
                    "source_id": source_id,
                    "chunk_index": idx,
                    "doc_type": metadata.get("doc_type", ""),
                    "filename": metadata.get("filename", ""),
                    "ingested_at": ingested_at,
                },
            }
            for idx, (text, emb) in enumerate(zip(chunks, embeddings))
        ]

        # Delete existing chunks for this source before re-ingesting
        await vector_store.delete_by_source(source_id)
        await vector_store.add_documents(chunk_dicts)

        span_data["chunk_count"] = len(chunks)
        span_data["status"] = "success"

    return {
        "source_id": source_id,
        "filename": metadata.get("filename", ""),
        "chunk_count": len(chunks),
        "doc_type": metadata.get("doc_type", ""),
    }
