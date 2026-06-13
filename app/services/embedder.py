import logging
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class EmbedderService:

    def __init__(self):
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        self._model = TextEmbedding(MODEL_NAME)
        logger.info("Embedding model loaded successfully")

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Convert a list of strings into a list of embedding vectors.
        Processes in batches for memory efficiency.

        Args:
            texts: List of strings to embed
            batch_size: How many strings to embed at once

        Returns:
            List of float vectors, one per input string
        """
        if not texts:
            return []

        embeddings = list(self._model.embed(texts, batch_size=batch_size))
        logger.debug(f"Embedded {len(texts)} texts → shape: ({len(embeddings)}, {len(embeddings[0])})")
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, query: str) -> list[float]:
        results = self.embed_texts([query])
        return results[0]


embedder = EmbedderService()