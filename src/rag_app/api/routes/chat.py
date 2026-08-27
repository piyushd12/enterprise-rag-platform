"""
Chat endpoint — invokes the LangGraph RAG pipeline.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends

from rag_app.api.dependencies import (
    get_embedding_provider,
    get_llm_provider,
    get_obs,
    get_vector_store,
)
from rag_app.api.schemas import ChatRequest, ChatResponse, SourceChunk
from rag_app.core.interfaces import EmbeddingProvider, LLMProvider, VectorStore
from rag_app.observability.provider import ObservabilityProvider
from rag_app.rag.graph import build_rag_graph

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    obs: ObservabilityProvider = Depends(get_obs),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_store: VectorStore = Depends(get_vector_store),
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> ChatResponse:
    """Ask a question against the ingested documents.

    Invokes the LangGraph RAG pipeline with LangSmith tracing.
    The request_id is passed as both a Logfire span attribute and
    LangSmith run metadata for cross-correlation.
    """
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    with obs.span("chat.request", {"request_id": request_id, "query": body.query}):
        # Build and invoke the LangGraph pipeline
        graph = build_rag_graph(embedding_provider, vector_store, llm_provider)

        # LangSmith tracing config — request_id in metadata for correlation
        config = {
            "metadata": {
                "request_id": request_id,
                "query": body.query,
            },
            "run_name": "rag_pipeline",
        }

        initial_state = {
            "request_id": request_id,
            "query": body.query,
            "top_k": body.top_k,
            "filters": body.filters,
        }

        result = await graph.ainvoke(initial_state, config=config)

        total_latency = round((time.perf_counter() - start) * 1000, 2)

        # Build source chunks for the response
        chunks = result.get("reranked_chunks") or result.get("retrieved_chunks", [])
        sources = [
            SourceChunk(
                content=c.content[:500],  # truncate for response
                source_id=c.metadata.get("source_id", ""),
                chunk_index=c.metadata.get("chunk_index", 0),
                score=round(c.score, 4),
                filename=c.metadata.get("filename", ""),
            )
            for c in chunks
        ]

    return ChatResponse(
        request_id=request_id,
        answer=result.get("generated_answer", ""),
        sources=sources,
        cached=result.get("cache_hit", False),
        llm_provider=result.get("llm_provider_used", ""),
        llm_model=result.get("llm_model_used", ""),
        latency_ms=total_latency,
    )
