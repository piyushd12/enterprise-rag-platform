"""
RAG pipeline state definition for LangGraph.

This TypedDict defines the shared state that flows through every node
in the RAG state graph. Fields are designed to be additive so that
new nodes (rerank, HyDE) can be inserted without changing existing ones.
"""

from __future__ import annotations

from typing import TypedDict

from rag_app.core.interfaces import RetrievedChunk


class RAGState(TypedDict, total=False):
    """State flowing through the LangGraph RAG pipeline.

    All fields are optional (total=False) so nodes only need to return
    the fields they modify.
    """

    # Input
    request_id: str
    query: str
    top_k: int
    filters: dict | None

    # Embedding
    query_embedding: list[float]

    # Cache
    cache_hit: bool
    cached_answer: str | None

    # Retrieval
    retrieved_chunks: list[RetrievedChunk]

    # Reranking (Phase 3 — pass-through for now)
    reranked_chunks: list[RetrievedChunk]

    # Generation
    generated_answer: str
    llm_provider_used: str
    llm_model_used: str
    llm_latency_ms: float
    total_tokens: int
