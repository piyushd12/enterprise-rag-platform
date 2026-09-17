"""
Chat endpoint — invokes the LangGraph RAG pipeline.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, Request

from rag_app.api.dependencies import (
    get_cache,
    get_embedding_provider,
    get_keyword_search,
    get_llm_provider,
    get_obs,
    get_reranker,
    get_vector_store,
)
from rag_app.api.rate_limit import limiter
from rag_app.api.schemas import ChatRequest, ChatResponse, SourceChunk
from rag_app.core.interfaces import (
    EmbeddingProvider,
    KeywordSearchProvider,
    LLMProvider,
    Reranker,
    VectorStore,
)
from rag_app.observability.provider import ObservabilityProvider
from rag_app.rag.graph import build_rag_graph

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    obs: ObservabilityProvider = Depends(get_obs),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_store: VectorStore = Depends(get_vector_store),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    cache=Depends(get_cache),
    reranker: Reranker = Depends(get_reranker),
    keyword_search: KeywordSearchProvider = Depends(get_keyword_search),
) -> ChatResponse:
    """Ask a question against the ingested documents.

    Invokes the LangGraph RAG pipeline with LangSmith tracing.
    The request_id is passed as both a Logfire span attribute and
    LangSmith run metadata for cross-correlation.

    Retrieval uses the cross-encoder reranker + BM25 hybrid search by
    default (verified in Phase 3 eval to fix real retrieval misses with
    no regressions), at the cost of added per-request latency (~3.5s+ for
    the reranker's cross-encoder scoring over a widened candidate pool).
    """
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    with obs.span("chat.request", {"request_id": request_id, "query": body.query}):
        # Build and invoke the LangGraph pipeline (with cache-aside)
        graph = build_rag_graph(
            embedding_provider,
            vector_store,
            llm_provider,
            cache=cache,
            reranker=reranker,
            keyword_search=keyword_search,
        )

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

        cache_hit = result.get("cache_hit", False)

        # On cache hit, answer comes from cached_answer; sources from cached data
        if cache_hit:
            answer = result.get("cached_answer", "")
            # Rebuild SourceChunk objects from cached source metadata
            cached_sources = result.get("cached_sources", [])
            sources = [
                SourceChunk(
                    content=s.get("content", ""),
                    source_id=s.get("source_id", ""),
                    chunk_index=s.get("chunk_index", 0),
                    score=s.get("score", 0.0),
                    filename=s.get("filename", ""),
                )
                for s in cached_sources
            ]
        else:
            answer = result.get("generated_answer", "")
            chunks = result.get("reranked_chunks") or result.get("retrieved_chunks", [])
            sources = [
                SourceChunk(
                    content=c.content[:500],
                    source_id=c.metadata.get("source_id", ""),
                    chunk_index=c.metadata.get("chunk_index", 0),
                    score=round(c.score, 4),
                    filename=c.metadata.get("filename", ""),
                )
                for c in chunks
            ]

    return ChatResponse(
        request_id=request_id,
        answer=answer,
        sources=sources,
        cached=cache_hit,
        llm_provider=result.get("llm_provider_used", "cache") if not cache_hit else "cache",
        llm_model=result.get("llm_model_used", "") if not cache_hit else "",
        latency_ms=total_latency,
    )
