"""
Abstract interfaces for the RAG application.

All concrete implementations depend on these ABCs. Route handlers, UI code,
and orchestration logic import ONLY from this module — never from concrete
SDK-specific implementations. This enforces the Dependency Inversion Principle
and makes every component independently testable and swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

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
    async def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        """Generate a completion for the given prompt.

        `system`, when given, is sent as a separate system message ahead of
        `prompt` (sent as the user message). Trusted instructions belong in
        `system`; `prompt` may carry untrusted text (retrieved chunks, chat
        history, the user's question), and keeping the two in separate
        messages is what lets the model tell instructions from data.
        Optional so callers that only send one string keep working.
        """
        ...

    @abstractmethod
    def stream(
        self, prompt: str, meta: dict[str, Any], system: str | None = None
    ) -> AsyncIterator[str]:
        """Stream a completion token-by-token for the given prompt.

        `system` works as in generate().

        Fills `meta` in place (provider, model, latency_ms, tokens_used)
        as streaming finishes -- a plain dict rather than a return value
        because an async generator's `return` can't carry a value out to
        an `async for` caller, and `meta` is caller-owned per call so
        there's no shared-state race across concurrent requests.
        """
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

    @abstractmethod
    async def list_sources(self) -> list[dict[str, Any]]:
        """List every distinct source document currently stored, aggregated
        by source_id (filename, doc_type, ingested_at, chunk_count)."""
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


class ChatStore(ABC):
    """Interface for persisted multi-turn chat sessions (separate from the
    answer cache -- this is durable conversation history, not a TTL'd
    performance optimization)."""

    @abstractmethod
    async def create_chat(self, title: str) -> str:
        """Create a new chat with the given title, returning its id."""
        ...

    @abstractmethod
    async def list_chats(self) -> list[dict[str, Any]]:
        """List all chats as {id, title, updated_at}, most recent first."""
        ...

    @abstractmethod
    async def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        """Get a chat's {id, title, messages}, or None if it doesn't exist."""
        ...

    @abstractmethod
    async def append_message(self, chat_id: str, message: dict[str, Any]) -> None:
        """Append one message ({role, content, ...}) to a chat's history.
        No-op if the chat doesn't exist (e.g. deleted mid-stream)."""
        ...

    @abstractmethod
    async def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and its history. Returns True if it existed."""
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


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

GuardAction = Literal["pass", "flag", "block"]


@dataclass
class GuardResult:
    """Outcome of one guardrail check.

    A block is a normal result, not an exception: a refused request still
    flows through the same response, persistence and logging code as an
    answered one, which is what keeps the two chat paths from drifting.
    `reason` is for logs and the API's `guardrail` field; `message` is the
    user-facing refusal and is only set on blocks.
    """

    action: GuardAction
    check: str  # e.g. "input.injection", "context.relevance", "output.leakage"
    reason: str = ""
    score: float | None = None
    message: str = ""


def first_block(results: list[GuardResult]) -> GuardResult | None:
    """The first blocking result, or None if nothing blocked."""
    return next((r for r in results if r.action == "block"), None)


class StreamFilter(ABC):
    """Screens LLM output on its way to the client, one token at a time.

    Streamed text can't be taken back once sent, so masking and leakage
    checks have to run before a token leaves the server, not after the
    answer is complete. The filter holds back a short tail of the stream
    and only releases text that can no longer turn into a match. Checks
    that need the whole answer (citations) run afterwards via
    Guardrails.review_output() and can only flag, never block.
    """

    results: list[GuardResult]

    @abstractmethod
    def feed(self, token: str) -> str:
        """Add a token; return the text now safe to send ("" to hold back).

        After a block (see first_block(self.results)) every further call
        returns "" -- the caller should stop consuming the LLM stream.
        """
        ...

    @abstractmethod
    def flush(self) -> str:
        """End of stream: return the held-back tail, screened."""
        ...


class Guardrails(ABC):
    """Every guardrail check for one request, behind a single interface.

    Both chat paths call these same methods -- /chat through LangGraph nodes
    (rag/nodes.py), /chat/stream directly -- so a check exists in exactly
    one place and can't silently cover one path but not the other.
    Implementations log every decision with the request_id themselves.
    """

    @abstractmethod
    async def check_input(
        self, query: str, history: list[dict[str, Any]], request_id: str = ""
    ) -> tuple[str, list[dict[str, Any]], list[GuardResult]]:
        """Screen the user's query and the chat history before anything
        else runs (cache lookup included).

        Returns the query to use from here on (secrets masked), the history
        to feed generation (turns that tripped a check removed), and the
        results.
        """
        ...

    @abstractmethod
    def check_context(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        reranked: bool,
        has_history: bool,
        request_id: str = "",
    ) -> tuple[list[RetrievedChunk], list[GuardResult]]:
        """Screen retrieved chunks between rerank and generate.

        Returns the chunks generation may see and the results; a block
        means generation should not run at all (no LLM call).
        """
        ...

    @abstractmethod
    def scan_chunk(self, text: str) -> tuple[str | None, list[str]]:
        """Ingestion-time scan of one chunk. Returns the text to store
        (None to drop the chunk) and flags for its payload's guard_flags."""
        ...

    @abstractmethod
    def stream_filter(self, request_id: str = "") -> StreamFilter:
        """A fresh output filter for one answer (streamed or not)."""
        ...

    @abstractmethod
    async def review_output(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        query: str,
        request_id: str = "",
    ) -> list[GuardResult]:
        """Checks that need the complete answer (citations, optional
        groundedness judge). Flags only -- by now the answer may already
        have been streamed."""
        ...

    @abstractmethod
    def redact(self, text: str) -> str:
        """Mask PII and secrets in text bound for logs and traces."""
        ...
