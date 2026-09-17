"""
Cross-encoder reranker using FastEmbed's local ONNX reranking models.

Uses ms-marco-MiniLM-L-6-v2 (~80MB) rather than a heavier reranker or a
separate sentence-transformers/torch dependency -- this project already
depends on fastembed for embeddings, and the model runs on the same local
ONNX runtime with no API key and a small memory footprint, which matters
on this project's memory-constrained host.
"""

from __future__ import annotations

from dataclasses import replace

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag_app.core.interfaces import Reranker, RetrievedChunk

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(Reranker):
    """Reranker implementation backed by a local cross-encoder ONNX model."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Score each chunk against the query and return the best top_n, ranked.

        The returned chunks' .score is overwritten with the cross-encoder's
        own score, not whatever score they carried in (candidates can come
        from dense search -- bounded cosine similarity -- or BM25 --
        unbounded term-frequency scores -- so those two are not comparable
        to begin with). After reranking, every returned chunk was scored by
        the same model on the same scale, which is both consistent and a
        more accurate relevance signal than the original retrieval score.
        Returns copies -- the input chunks (e.g. the wider pre-rerank
        candidate pool) are left untouched.
        """
        if not chunks:
            return []

        documents = [c.content for c in chunks]
        scores = list(self._model.rerank(query, documents))

        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [replace(chunk, score=float(score)) for chunk, score in ranked[:top_n]]
