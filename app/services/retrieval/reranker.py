# app/services/retrieval/reranker.py
"""
Cross-encoder re-ranking service.

Week 6 upgrade: hybrid retrieval returns the top-30 candidate chunks,
the cross-encoder then re-ranks them and returns the best top-k.

Why cross-encoders outperform bi-encoders for final ranking:
- Bi-encoders (fastembed): encode query and chunk INDEPENDENTLY, then compare.
  Fast, but loses the interaction signal between the two texts.
- Cross-encoders: encode query AND chunk TOGETHER in one pass.
  Slower (O(n) forward passes per query), but far more accurate — the model
  can attend to both texts simultaneously, catching subtle relevance signals
  that bi-encoders miss entirely.

Typical use: retrieve top-30 with hybrid search (fast), re-rank to top-5
with cross-encoder (slower but only 30 forward passes), pass to LLM.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 6-layer MiniLM fine-tuned on MS MARCO passage ranking
  - ~66MB — smallest high-quality cross-encoder available
  - ~100ms for 30 candidates on CPU, ~20ms on GPU
"""
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Model stored in the project's .cache dir to avoid re-downloads
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CACHE_DIR = str(Path(__file__).parent.parent.parent.parent / ".cache" / "cross_encoder")


class RerankerService:
    """
    Cross-encoder re-ranker using sentence-transformers CrossEncoder.

    Loaded lazily on first use — avoids slowing down server startup
    with a 66MB model download if reranking is not configured.
    """

    def __init__(self, model_name: str = _MODEL_NAME, cache_dir: str = _CACHE_DIR):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self._model_name}")
            self._model = CrossEncoder(
                self._model_name,
                cache_folder=self._cache_dir,
                max_length=512,
            )
            logger.info("Cross-encoder loaded ✅")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            raise RuntimeError(
                f"Cross-encoder model could not be loaded: {e}"
            ) from e

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank a list of candidate chunks using cross-encoder scores.

        Args:
            query: The user's search query (post-rewriting).
            chunks: Candidate chunks from hybrid retrieval (already pre-filtered
                    to ~20-30 best candidates). Must have 'chunk_text' field.
            top_k: How many top chunks to return after re-ranking.

        Returns:
            Re-ranked list (best first), limited to top_k.
            Each chunk gets a 'rerank_score' field (raw cross-encoder logit).
        """
        if not chunks:
            return []

        if len(chunks) == 1:
            # Nothing to rank with a single chunk
            return chunks

        self._ensure_loaded()

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, chunk["chunk_text"]) for chunk in chunks]

        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.warning(f"Cross-encoder predict failed: {e} — returning hybrid results")
            return chunks[:top_k]

        # Attach scores and sort descending
        scored = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = float(score)
            scored.append(chunk_copy)

        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        top = scored[:top_k]

        logger.info(
            f"Reranked {len(chunks)} → {len(top)} chunks. "
            f"Top score: {top[0]['rerank_score']:.3f}, "
            f"Bottom score: {top[-1]['rerank_score']:.3f}"
        )
        return top

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# Module-level singleton — loaded lazily on first rerank() call
reranker = RerankerService()
