"""
FastEmbed-based embedding provider.

Uses Qdrant's FastEmbed library for local, free, no-API-key embeddings.
Default model: BAAI/bge-small-en-v1.5 (384 dimensions).
"""

from __future__ import annotations

from fastembed import TextEmbedding

from rag_app.config.settings import settings
from rag_app.core.interfaces import EmbeddingProvider


# Model name → vector dimension mapping for supported models
_MODEL_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class FastEmbedProvider(EmbeddingProvider):
    """EmbeddingProvider implementation backed by FastEmbed (ONNX, local CPU)."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.embedding_model
        self._model = TextEmbedding(model_name=self._model_name)
        self._dimension = _MODEL_DIMENSIONS.get(self._model_name, 384)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        embeddings = list(self._model.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()

    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._dimension
