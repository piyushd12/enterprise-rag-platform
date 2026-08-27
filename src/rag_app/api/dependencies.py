"""
FastAPI dependency injection factories.

Provides shared instances of providers via Depends() so that
route handlers never construct providers directly. All providers
are singletons scoped to the app lifespan.
"""

from __future__ import annotations

from functools import lru_cache

from rag_app.core.interfaces import EmbeddingProvider, LLMProvider, VectorStore
from rag_app.embeddings.fastembed_provider import FastEmbedProvider
from rag_app.llm.groq_provider import GroqProvider
from rag_app.llm.openrouter_provider import OpenRouterProvider
from rag_app.llm.router import LLMRouter
from rag_app.observability.provider import ObservabilityProvider
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
