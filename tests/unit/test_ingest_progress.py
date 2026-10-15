"""Unit test: ingest_file reports embedding progress batch by batch."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from rag_app.ingestion import pipeline


@contextmanager
def _span(*_args, **_kwargs):
    yield {}


async def test_ingest_reports_progress_per_batch():
    chunks = [f"chunk {i}" for i in range(150)]
    embedder = MagicMock()
    embedder.embed_documents.side_effect = lambda batch: [[0.0] for _ in batch]
    store = MagicMock(delete_by_source=AsyncMock(), add_documents=AsyncMock())
    obs = MagicMock(span=_span)
    doc = {"content": "x", "metadata": {"source_id": "s1", "filename": "f.txt", "doc_type": "txt"}}
    calls: list[tuple[int, int]] = []

    with (
        patch.object(pipeline, "load_document", return_value=[doc]),
        patch.object(pipeline, "chunk_text", return_value=chunks),
    ):
        result = await pipeline.ingest_file(
            "f.txt", embedder, store, obs, on_progress=lambda d, t: calls.append((d, t))
        )

    assert calls == [(0, 150), (64, 150), (128, 150), (150, 150)]
    assert result["chunk_count"] == 150
    assert len(store.add_documents.call_args.args[0]) == 150
