"""
BM25 keyword search over the ingested corpus.

Complements dense vector search: dense embeddings capture semantic
similarity but can under-weight distinctive exact terms (proper nouns,
specific phrases), which BM25's term-frequency/IDF weighting rewards
directly. Verified empirically on a real failure case: a passage dense
search ranked ~50th out of 764 candidates for a paraphrased question,
BM25 ranked it 2nd, because the passage contains rare, exact terms
("three miles", "Netherfield") that appear in the query almost verbatim.

Used as a second candidate source for the reranker (see rag/nodes.py),
not a replacement for dense search -- the cross-encoder reranker sees
the union of both and picks the final top_k, so the LLM's context size
is unaffected either way.

The index is built once (lazily, on first use) from every chunk
currently in the vector store and cached for the process lifetime. It
does not automatically refresh when new documents are ingested --
restart the process (or call rebuild()) to pick up newly ingested
chunks. Acceptable for this project's scope; a production system would
rebuild on ingestion the same way the Redis answer cache's
collection_version counter invalidates on ingestion.
"""

from __future__ import annotations

import re
import time

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from rag_app.core.interfaces import KeywordSearchProvider, RetrievedChunk
from rag_app.observability.provider import ObservabilityProvider

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index(KeywordSearchProvider):
    """In-memory BM25 index built from a Qdrant collection's stored chunks."""

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        obs: ObservabilityProvider | None = None,
    ) -> None:
        self._client = QdrantClient(url=qdrant_url)
        self._collection_name = collection_name
        self._obs = obs
        self._bm25: BM25Okapi | None = None
        self._chunks: list[RetrievedChunk] = []

    def rebuild(self) -> None:
        """(Re)build the index from the collection's current contents.

        Logged to Logfire: this scrolls the entire collection and
        tokenizes every chunk, which isn't free, and previously ran
        invisibly -- most notably on the lazy first-search path (see
        search() below), which fires on every eval run since a fresh
        BM25Index is constructed per run and never explicitly rebuilt
        ahead of time, silently inflating the retrieve node's latency
        with no way to tell rebuild cost apart from actual search cost.
        """
        start = time.perf_counter()
        points, _ = self._client.scroll(
            collection_name=self._collection_name,
            limit=10_000,
            with_payload=True,
        )
        self._chunks = [
            RetrievedChunk(content=p.payload.get("content", ""), score=0.0, metadata=p.payload)
            for p in points
            if p.payload.get("content")
        ]
        tokenized = [_tokenize(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

        if self._obs:
            self._obs.log_event(
                "bm25.index.rebuilt",
                {
                    "chunk_count": len(self._chunks),
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top_k chunks by BM25 score for the query."""
        if self._bm25 is None:
            self.rebuild()
        if self._bm25 is None or not self._chunks:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                content=self._chunks[i].content,
                score=float(scores[i]),
                metadata=self._chunks[i].metadata,
            )
            for i in ranked_idx
        ]
