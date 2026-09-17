"""
FastAPI dependency injection factories.

Provides shared instances of providers via Depends() so that
route handlers never construct providers directly. All providers
are singletons scoped to the app lifespan.
"""

from __future__ import annotations

from functools import lru_cache

from rag_app.config.settings import settings
from rag_app.core.interfaces import (
    EmbeddingProvider,
    KeywordSearchProvider,
    LLMProvider,
    Reranker,
    TaskQueue,
    VectorStore,
)
from rag_app.embeddings.fastembed_provider import FastEmbedProvider
from rag_app.llm.groq_provider import GroqProvider
from rag_app.llm.openrouter_provider import OpenRouterProvider
from rag_app.llm.router import LLMRouter
from rag_app.observability.provider import ObservabilityProvider
from rag_app.reranking.cross_encoder_reranker import CrossEncoderReranker
from rag_app.retrieval.bm25_index import BM25Index
from rag_app.vectorstore.qdrant_store import QdrantStore


@lru_cache
def get_obs() -> ObservabilityProvider:
    """Singleton ObservabilityProvider."""
    return ObservabilityProvider()


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Singleton EmbeddingProvider (FastEmbed)."""
    return FastEmbedProvider()


@lru_cache
def get_vector_store() -> VectorStore:
    """Singleton VectorStore (Qdrant)."""
    obs = get_obs()
    embedder = get_embedding_provider()
    return QdrantStore(obs=obs, embedding_dimension=embedder.dimension())


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Singleton LLM Router (Groq → OpenRouter fallback)."""
    obs = get_obs()
    return LLMRouter(
        primary=GroqProvider(),
        fallback=OpenRouterProvider(),
        obs=obs,
    )


@lru_cache
def get_reranker() -> Reranker:
    """Singleton cross-encoder Reranker (local ONNX, no API cost).

    Wired into /chat by default (see routes/chat.py) and available for
    the eval harness's use_reranker flag.
    """
    return CrossEncoderReranker()


@lru_cache
def get_keyword_search() -> KeywordSearchProvider:
    """Singleton BM25 keyword search index over the vector store's chunks.

    Only meaningful combined with get_reranker() -- BM25 is a candidate
    source for the reranker, not a standalone final ranking. Builds lazily
    on first search() call; does not auto-refresh on new ingestion (see
    BM25Index docstring).
    """
    return BM25Index(
        qdrant_url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        obs=get_obs(),
    )


@lru_cache
def get_cache():
    """Singleton RedisCache provider (cache-aside pattern)."""
    from rag_app.caching.redis_cache import RedisCache
    obs = get_obs()
    return RedisCache(obs=obs)


@lru_cache
def get_queue() -> TaskQueue:
    """Singleton TaskQueue (Celery-backed)."""
    from rag_app.queue.redis_queue import RedisQueue
    obs = get_obs()
    return RedisQueue(obs=obs)
