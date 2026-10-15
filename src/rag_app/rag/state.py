"""
RAG pipeline state definition for LangGraph.

This TypedDict defines the shared state that flows through every node
in the RAG state graph. Fields are designed to be additive so that
new nodes (rerank, HyDE) can be inserted without changing existing ones.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from rag_app.core.interfaces import GuardResult, RetrievedChunk


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
    # Prior chat turns fed to generation ({role, content, ...} dicts). Empty
    # for /chat and eval; the input guard returns it with excluded turns
    # removed.
    history: list[dict]

    # Guardrails: every check's result for this request. A reducer field
    # (operator.add), so each guard node appends its results instead of
    # overwriting the previous node's.
    guard_results: Annotated[list[GuardResult], operator.add]

    # Embedding
    query_embedding: list[float]

    # HyDE (Phase 3 — optional, off by default)
    hyde_document: str | None

    # Cache
    cache_hit: bool
    cached_answer: str | None
    cached_sources: list[dict]

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
