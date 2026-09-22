"""
Chat endpoint — invokes the LangGraph RAG pipeline.
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from rag_app.api.dependencies import (
    get_cache,
    get_chat_store,
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
    ChatStore,
    EmbeddingProvider,
    KeywordSearchProvider,
    LLMProvider,
    Reranker,
    VectorStore,
)
from rag_app.observability.provider import ObservabilityProvider
from rag_app.rag.graph import build_rag_graph
from rag_app.rag.nodes import _build_rag_prompt, build_context, make_rerank_node, make_retrieve_node

router = APIRouter(tags=["chat"])


def _sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


async def _generate_chat_title(llm_provider: LLMProvider, query: str) -> str:
    """Summarize a chat's first question into a short title, ChatGPT/Claude-
    style. Falls back to a truncated query on any LLM failure -- title
    generation must never be the reason a chat fails to start."""
    prompt = (
        "Summarize the following user question into a short chat title "
        '(3-6 words, no quotes, no trailing punctuation).\n\nQuestion: "'
        f'{query}"\n\nTitle:'
    )
    try:
        response = await llm_provider.generate(prompt)
        title = response.content.strip().strip('"').strip("'")
        return title[:60] or query[:60]
    except Exception:
        return query[:60]


def _format_history(messages: list[dict], limit: int = 6) -> str:
    """Render the last `limit` stored messages as a transcript block, so
    generation can resolve follow-ups against earlier turns in the same
    chat. Empty string (no block at all) when there's no prior history."""
    recent = messages[-limit:]
    if not recent:
        return ""
    lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent]
    return "Previous conversation:\n" + "\n".join(lines) + "\n"


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


@router.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    obs: ObservabilityProvider = Depends(get_obs),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_store: VectorStore = Depends(get_vector_store),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    cache=Depends(get_cache),
    reranker: Reranker = Depends(get_reranker),
    keyword_search: KeywordSearchProvider = Depends(get_keyword_search),
    chat_store: ChatStore = Depends(get_chat_store),
) -> StreamingResponse:
    """Same pipeline as /chat, streamed as Server-Sent Events, within a
    persisted chat session.

    Cache lookup and retrieval/reranking reuse the exact same node
    factory functions /chat uses. Only generation differs: tokens are
    streamed to the client as the LLM produces them instead of waiting
    for the full answer, cutting perceived latency during the multi-second
    generation phase. A cache hit still returns instantly as one "answer"
    event rather than being artificially drawn out token-by-token.

    Bypasses the compiled LangGraph (calls the node functions directly)
    since a graph node returns one state update, not a token stream --
    the individual LLM call is still traced by LangSmith on its own
    (that instrumentation lives on the LangChain chat model itself), just
    not nested under a single parent "rag_pipeline" run the way /chat's
    graph.ainvoke() is.

    Chat sessions: if `body.chat_id` is omitted (or unknown), a new chat
    is created with a title generated from this first question (one extra
    short LLM call, made synchronously before generation starts -- so new
    chats pay a small latency cost existing chats don't). Every message is
    persisted to the chat's history, and recent turns from that history
    are fed back into the generation prompt so follow-up questions can be
    resolved against what was already discussed. Retrieval itself is not
    history-aware -- only generation sees prior turns.
    """
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    async def event_stream():
        with obs.span("chat.stream.request", {"request_id": request_id, "query": body.query}):
            existing_chat = await chat_store.get_chat(body.chat_id) if body.chat_id else None
            history_messages = existing_chat["messages"] if existing_chat else []
            if existing_chat:
                chat_id = body.chat_id
            else:
                title = await _generate_chat_title(llm_provider, body.query)
                chat_id = await chat_store.create_chat(title)

            await chat_store.append_message(
                chat_id, {"role": "user", "content": body.query, "ts": time.time()}
            )

            cached = await cache.get_answer(body.query) if cache else None
            if cached is not None:
                await chat_store.append_message(
                    chat_id,
                    {
                        "role": "assistant",
                        "content": cached["answer"],
                        "sources": cached.get("sources", []),
                        "ts": time.time(),
                    },
                )
                yield _sse("answer", {"content": cached["answer"]})
                yield _sse(
                    "done",
                    {
                        "request_id": request_id,
                        "chat_id": chat_id,
                        "sources": cached.get("sources", []),
                        "cached": True,
                        "llm_provider": "cache",
                        "llm_model": "",
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    },
                )
                return

            retrieve = make_retrieve_node(
                embedding_provider,
                vector_store,
                use_reranker=reranker is not None,
                keyword_search=keyword_search,
            )
            state: dict = {"query": body.query, "top_k": body.top_k, "filters": body.filters}
            state.update(await retrieve(state))
            if reranker is not None:
                state.update(await make_rerank_node(reranker)(state))

            chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])
            history = _format_history(history_messages)
            prompt = _build_rag_prompt(body.query, build_context(chunks), history)

            meta: dict = {}
            full_answer = ""
            try:
                async for token in llm_provider.stream(prompt, meta):
                    full_answer += token
                    yield _sse("token", {"content": token})
            except Exception as exc:
                yield _sse("error", {"message": str(exc)[:500]})
                return

            sources = [
                {
                    "content": c.content[:500],
                    "source_id": c.metadata.get("source_id", ""),
                    "chunk_index": c.metadata.get("chunk_index", 0),
                    "score": round(c.score, 4),
                    "filename": c.metadata.get("filename", ""),
                }
                for c in chunks
            ]
            if cache:
                await cache.set_answer(body.query, full_answer, sources)

            await chat_store.append_message(
                chat_id,
                {
                    "role": "assistant",
                    "content": full_answer,
                    "sources": sources,
                    "llm_provider": meta.get("provider", ""),
                    "llm_model": meta.get("model", ""),
                    "ts": time.time(),
                },
            )

            yield _sse(
                "done",
                {
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "sources": sources,
                    "cached": False,
                    "llm_provider": meta.get("provider", ""),
                    "llm_model": meta.get("model", ""),
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
