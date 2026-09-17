"""
Individual node functions for the RAG LangGraph pipeline.

Each node is a pure function: takes RAGState, returns a partial state update.
Nodes are independently testable and traced by LangSmith as child runs.

Node topology (Phase 2):
    cache_lookup → [hit?] END
                   [miss] retrieve → generate → cache_write → END
"""

from __future__ import annotations

from rag_app.caching.redis_cache import RedisCache
from rag_app.core.interfaces import (
    EmbeddingProvider,
    KeywordSearchProvider,
    LLMProvider,
    Reranker,
    RetrievedChunk,
    VectorStore,
)
from rag_app.rag.state import RAGState

# How much wider than the final top_k to cast the retrieval net when a
# reranker is active, and the hard ceiling on that width regardless of
# top_k. A bi-encoder's independent query/chunk embeddings are a cheap but
# imprecise similarity signal -- widening the candidate pool gives the
# (more accurate, jointly-scored) cross-encoder reranker room to find a
# correct chunk that the bi-encoder ranked outside a naive top-5 cutoff.
RERANK_CANDIDATE_MULTIPLIER = 10
RERANK_CANDIDATE_MAX = 50


def make_hyde_expand_node(llm_provider: LLMProvider):
    """Factory for the HyDE (Hypothetical Document Embeddings) node.

    Generates a short hypothetical passage that would plausibly answer the
    query, as if drawn directly from a reference document. Embedding this
    passage instead of the raw question closes the vocabulary gap between
    question-style queries (e.g. "what is the opening line of X") and
    narrative/technical prose that never restates the question -- the
    retrieve node uses hyde_document for embedding when present, falling
    back to the raw query otherwise.
    """

    async def hyde_expand(state: RAGState) -> dict:
        query = state["query"]
        prompt = _build_hyde_prompt(query)
        response = await llm_provider.generate(prompt)
        return {"hyde_document": response.content}

    return hyde_expand


def make_retrieve_node(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    use_reranker: bool = False,
    keyword_search: KeywordSearchProvider | None = None,
):
    """Factory for the retrieve node — closes over providers.

    When use_reranker is True, searches a wider candidate pool than the
    requested top_k (see RERANK_CANDIDATE_MULTIPLIER) so the rerank node
    has room to promote a correct chunk the bi-encoder under-ranked.
    reranked_chunks still passes through the raw (wide) results here --
    the rerank node overwrites it with the trimmed, re-scored top_k.

    When keyword_search is also provided (only meaningful alongside a
    reranker -- otherwise nothing re-scores the merged set), BM25 keyword
    search runs alongside dense search and the two candidate pools are
    merged (deduped by source_id + chunk_index) before reranking. This
    catches passages containing distinctive exact terms that dense
    embeddings under-weight, which a wider dense-only search alone won't
    reliably surface.
    """

    async def retrieve(state: RAGState) -> dict:
        """Embed the query (or HyDE passage, if present) and search the vector store."""
        query = state["query"]
        top_k = state.get("top_k", 5)
        filters = state.get("filters")

        search_k = (
            min(top_k * RERANK_CANDIDATE_MULTIPLIER, RERANK_CANDIDATE_MAX)
            if use_reranker
            else top_k
        )

        # When a HyDE passage is available, blend its embedding with the
        # raw query's rather than replacing the query outright: the HyDE
        # passage reads more like the source documents (helps close the
        # question-vs-prose vocabulary gap), but it can also hallucinate a
        # confidently wrong specific detail, which would otherwise steer
        # retrieval entirely toward the wrong content. Averaging tempers
        # a bad HyDE passage with the real question instead of fully
        # trusting it. Both embeddings are local (FastEmbed), so this adds
        # no extra API cost.
        hyde_document = state.get("hyde_document")
        if hyde_document:
            query_embedding_only = embedding_provider.embed_query(query)
            hyde_embedding = embedding_provider.embed_query(hyde_document)
            query_embedding = [
                (a + b) / 2 for a, b in zip(query_embedding_only, hyde_embedding)
            ]
        else:
            query_embedding = embedding_provider.embed_query(query)

        # Search vector store
        chunks = await vector_store.search(
            query_vector=query_embedding,
            top_k=search_k,
            filters=filters,
        )

        if keyword_search is not None:
            bm25_chunks = keyword_search.search(query, top_k=search_k)
            chunks = _merge_dedup(chunks, bm25_chunks)

        return {
            "query_embedding": query_embedding,
            "retrieved_chunks": chunks,
            # Pass-through when no rerank node follows; the rerank node
            # (when present) overwrites this with the trimmed, re-scored set.
            "reranked_chunks": chunks[:top_k],
        }

    return retrieve


def _merge_dedup(
    primary: list[RetrievedChunk], secondary: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Union two candidate lists, deduped by (source_id, chunk_index).

    Chunks missing that metadata (shouldn't happen for ingested content,
    but keeps this robust) fall back to deduping on content instead.
    """

    def key(chunk: RetrievedChunk) -> tuple:
        meta = chunk.metadata
        if "source_id" in meta and "chunk_index" in meta:
            return (meta["source_id"], meta["chunk_index"])
        return (chunk.content,)

    seen = {key(c) for c in primary}
    merged = list(primary)
    for chunk in secondary:
        k = key(chunk)
        if k not in seen:
            seen.add(k)
            merged.append(chunk)
    return merged


def make_rerank_node(reranker: Reranker):
    """Factory for the rerank node — closes over the reranker.

    Re-scores the wide candidate pool retrieve produced with a cross-encoder
    (query and chunk jointly encoded, a much finer-grained relevance signal
    than the bi-encoder's independent embeddings) and trims back down to the
    originally requested top_k before generation ever sees it -- generation's
    input size is unaffected by how wide the candidate pool was.
    """

    async def rerank(state: RAGState) -> dict:
        query = state["query"]
        top_k = state.get("top_k", 5)
        candidates = state.get("retrieved_chunks", [])

        reranked = reranker.rerank(query, candidates, top_n=top_k)
        return {"reranked_chunks": reranked}

    return rerank


def make_generate_node(llm_provider: LLMProvider):
    """Factory for the generate node — closes over the LLM provider."""

    async def generate(state: RAGState) -> dict:
        """Generate an answer grounded in the retrieved context."""
        query = state["query"]
        # Use reranked chunks if available, otherwise retrieved
        chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])

        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("filename", chunk.metadata.get("source_id", "unknown"))
            context_parts.append(f"[Source {i + 1}: {source}]\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)

        # Build prompt
        prompt = _build_rag_prompt(query, context)

        # Generate
        response = await llm_provider.generate(prompt)

        return {
            "generated_answer": response.content,
            "llm_provider_used": response.provider,
            "llm_model_used": response.model,
            "llm_latency_ms": response.latency_ms,
            "total_tokens": response.tokens_used,
        }

    return generate


def make_cache_lookup_node(cache: RedisCache):
    """Factory for the cache-lookup node — checks for a cached answer.

    On cache hit: sets cache_hit=True, cached_answer with the stored content,
    and cached_sources with the stored source metadata.
    On cache miss: sets cache_hit=False so downstream nodes run normally.
    """
    async def cache_lookup(state: RAGState) -> dict:
        query = state["query"]
        result = await cache.get_answer(query)

        if result is not None:
            return {
                "cache_hit": True,
                "cached_answer": result["answer"],
                "cached_sources": result.get("sources", []),
            }
        return {"cache_hit": False}

    return cache_lookup


def make_cache_write_node(cache: RedisCache):
    """Factory for the cache-write node — stores generated answer after retrieval.

    Serializes the answer and source metadata as JSON and writes to the answer
    cache with the configured TTL. Called only on cache-miss paths.
    """
    async def cache_write(state: RAGState) -> dict:
        query = state["query"]
        answer = state.get("generated_answer", "")
        chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])

        # Serialize sources for cache (avoid storing full large content)
        sources_for_cache = [
            {
                "content": c.content[:500],
                "source_id": c.metadata.get("source_id", ""),
                "chunk_index": c.metadata.get("chunk_index", 0),
                "score": c.score,
                "filename": c.metadata.get("filename", ""),
            }
            for c in chunks
        ]
        await cache.set_answer(query, answer, sources_for_cache)
        return {}

    return cache_write


def _redirect_after_cache_lookup(state: RAGState) -> str:
    """Conditional edge: skip retrieve/generate/cache_write on cache hit."""
    if state.get("cache_hit"):
        return "end"
    return "retrieve"


def _build_hyde_prompt(query: str) -> str:
    """Build the HyDE prompt: ask for a plausible source passage, not an answer to the user.

    Deliberately does NOT ask the model to "write a plausible/generic passage"
    when uncertain -- that phrasing invites confident-sounding fabrication of
    specific facts (causes, motivations, relationships) it doesn't actually
    know, which then drives retrieval toward the wrong content entirely.
    Instead it asks for restatement of the question's subject matter in
    source-appropriate style, explicitly discouraging invented specifics.
    """
    return f"""Write a short passage (2-4 sentences) in the style of the kind of \
document this question is about (narrative prose for a novel, encyclopedic prose \
for a reference article, technical prose for a research paper, etc). Restate the \
question's subject and key terms as declarative statements, using vocabulary and \
phrasing typical of that source material.

Do NOT invent specific facts, causes, motivations, or plot details you are not \
confident are correct. If you don't know a specific detail the question asks for \
(e.g. why something happened, or a precise figure), describe the general topic \
and its context instead of guessing a specific answer -- an unconfident but \
honest passage is far more useful here than a confident but wrong one.

Do not address the reader, and do not mention that this is hypothetical.

Question: {query}

Passage:"""


def _build_rag_prompt(query: str, context: str) -> str:
    """Build the RAG prompt with context and query."""
    return f"""You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context below to answer the question.
If the context doesn't contain enough information to answer, say so clearly.
Cite the source numbers [Source N] when using information from the context.

Context:
{context}

Question: {query}

Answer:"""
