"""
Abstract interfaces for the RAG application.

All concrete implementations depend on these ABCs. Route handlers, UI code,
and orchestration logic import ONLY from this module — never from concrete
SDK-specific implementations. This enforces the Dependency Inversion Principle
and makes every component independently testable and swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    provider: str  # e.g. "groq", "openrouter"
    model: str
    tokens_used: int
    latency_ms: float


@dataclass
class RetrievedChunk:
    """A single chunk returned from vector search."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Interface for language model providers (Groq, OpenRouter, etc.)."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion for the given prompt."""
        ...


class EmbeddingProvider(ABC):
    """Interface for embedding models (FastEmbed, OpenAI, etc.)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts. May be called with large batches."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Optimized for single-text latency."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...


class VectorStore(ABC):
    """Interface for vector database operations."""

    @abstractmethod
    async def add_documents(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Upsert document chunks (with embeddings and metadata) into the store.

        Each chunk dict must contain:
        - 'id': str — unique point ID
        - 'embedding': list[float] — vector
        - 'content': str — raw text
        - 'metadata': dict — arbitrary payload fields
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Search for the top_k most similar chunks to the query vector."""
        ...

    @abstractmethod
    async def delete_by_source(self, source_id: str) -> None:
        """Delete all chunks associated with a given source document."""
        ...

    @abstractmethod
    async def ensure_collection(self) -> None:
        """Create the collection/index if it does not already exist."""
        ...


class CacheProvider(ABC):
    """Interface for key-value cache (Redis, in-memory, etc.)."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Retrieve a value by key. Returns None on cache miss."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with optional TTL in seconds."""
        ...

    @abstractmethod
    async def invalidate_pattern(self, pattern: str) -> int:
        """Delete all keys matching the glob pattern. Returns count deleted."""
        ...


class TaskQueue(ABC):
    """Interface for async task queue (Celery, RQ, etc.)."""

    @abstractmethod
    def enqueue(self, task_name: str, *args: Any, **kwargs: Any) -> str:
        """Enqueue a task and return its task ID immediately."""
        ...

    @abstractmethod
    def get_status(self, task_id: str) -> dict[str, Any]:
        """Poll the status of a previously enqueued task.

        Returns a dict with at least:
        - 'task_id': str
        - 'status': str (queued, started, succeeded, failed, retried)
        - 'result': Any | None
        - 'error': str | None
        """
        ...


class Reranker(ABC):
    """Interface for cross-encoder rerankers (jointly score query+chunk pairs).

    Intended to run over a wider candidate pool than the final generation
    context needs: a bi-encoder's independent query/chunk embeddings are a
    cheap but imprecise similarity signal, so retrieval widens the net and
    the reranker narrows it back down to top_n before anything reaches the
    LLM -- the LLM's context size never grows.
    """

    @abstractmethod
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Score each chunk against the query and return the best top_n, ranked."""
        ...


class KeywordSearchProvider(ABC):
    """Interface for lexical/keyword search (e.g. BM25) over the corpus.

    Complements dense vector search: independent query/chunk embeddings
    capture semantic similarity but can under-weight distinctive exact
    terms (proper nouns, specific phrases) that lexical scoring rewards
    directly. Used as a second candidate source alongside dense retrieval,
    with a Reranker choosing the final top_n from their combined pool.
    """

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top_k chunks by lexical relevance to the query."""
        ...
