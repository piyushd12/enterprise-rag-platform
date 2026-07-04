# app/services/rag/pipeline.py
"""
Full RAG pipeline: question → hybrid retrieval → context → LLM → answer.

Week 5 upgrade over Week 4:
- Hybrid retrieval (dense + BM25 → RRF) replaces dense-only search
- Query rewriting is now called before retrieval for multi-turn accuracy
- Top-k candidates are larger before fusion (20+20 → fused top 10)
"""
import logging

from app.services.retrieval.hybrid_retriever import hybrid_search
from app.services.rag.context_builder import build_context, format_history_for_llm
from app.services.rag.generator import generate_rag_response, rewrite_query_for_retrieval
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


async def workspace_has_chunked_documents(workspace_id: str) -> bool:
    """
    Quick pre-flight check before running the full RAG pipeline.
    Returns False if no chunked documents exist in this workspace.
    """
    from app.services.vector_store import vector_store, COLLECTION_NAME
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    count = vector_store.client.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="workspace_id",
                    match=MatchValue(value=workspace_id),
                )
            ]
        ),
        exact=False,
    )
    return count.count > 0


async def run_rag_pipeline(
    question: str,
    workspace_id: str,
    conversation_history: list,
    top_k: int = 10,
) -> dict:
    """
    Orchestrates the full retrieval-augmented generation pipeline.

    Pipeline steps:
      1. Format conversation history for the LLM
      2. Rewrite the question for retrieval (resolves pronouns, follow-ups)
      3. Hybrid retrieval: dense + BM25 → RRF fusion
      4. Pack retrieved chunks within the context token budget
      5. Call the LLM with the grounded context
      6. Return answer + structured source list

    Args:
        question: The user's raw question
        workspace_id: Tenant scope for retrieval
        conversation_history: List of Message ORM objects (multi-turn context)
        top_k: Final number of chunks after fusion and context packing

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "tokens_used": int | None,
            "chunks_retrieved": int,
            "chunks_used": int,
        }
    """
    logger.info(
        f"RAG pipeline started: workspace={workspace_id[:8]}, "
        f"question='{question[:60]}'"
    )

    # ── Step 1: Format conversation history ─────────────────────────────────
    history = format_history_for_llm(conversation_history)

    # ── Step 2: Query rewriting for multi-turn retrieval ────────────────────
    # Resolves references like "it", "that", "the first one" using the LLM.
    # Returns the question unchanged if it's already standalone.
    search_query = await rewrite_query_for_retrieval(question, history)
    if search_query.lower() != question.lower():
        logger.info(f"Query rewritten: '{question}' → '{search_query}'")

    # ── Step 3: Hybrid retrieval (dense + BM25 → RRF) ───────────────────────
    raw_chunks = await hybrid_search(
        query=search_query,
        workspace_id=workspace_id,
        top_k=top_k,
        dense_top_k=top_k * 2,   # retrieve 2× candidates from each method
        bm25_top_k=top_k * 2,    # before fusing down to top_k
    )

    if not raw_chunks:
        logger.warning(f"No chunks retrieved for: '{search_query[:60]}'")
        return {
            "answer": (
                "I couldn't find any relevant information in your documents "
                "to answer this question. Please make sure you have uploaded "
                "documents that contain information about this topic."
            ),
            "sources": [],
            "tokens_used": None,
            "chunks_retrieved": 0,
            "chunks_used": 0,
        }

    # ── Step 4: Pack context within token budget ─────────────────────────────
    context, used_chunks = build_context(
        chunks=raw_chunks,
        max_tokens=settings.rag_context_token_budget,
    )

    # ── Step 5: Call LLM ─────────────────────────────────────────────────────
    llm_result = await generate_rag_response(
        question=question,
        context=context,
        history=history,
    )

    # ── Step 6: Structure the source list for the API response ───────────────
    sources = [
        {
            "document_id": chunk["document_id"],
            "chunk_text": chunk["chunk_text"],
            "page_num": chunk["page_num"],
            "score": chunk["score"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in used_chunks
    ]

    logger.info(
        f"RAG pipeline complete: "
        f"retrieved={len(raw_chunks)}, used={len(used_chunks)}, "
        f"tokens={llm_result['tokens_used']}"
    )

    return {
        "answer": llm_result["answer"],
        "sources": sources,
        "tokens_used": llm_result["tokens_used"],
        "chunks_retrieved": len(raw_chunks),
        "chunks_used": len(used_chunks),
    }