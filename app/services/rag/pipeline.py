import logging

from app.services.embedder import embedder
from app.services.vector_store import vector_store
from app.services.rag.context_builder import build_context, format_history_for_llm
from app.services.rag.generator import generate_rag_response
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


async def run_rag_pipeline(
    question: str,
    workspace_id: str,
    conversation_history: list,
    top_k: int = 3,
) -> dict:
    logger.info(
        f"RAG pipeline started: workspace={workspace_id}, "
        f"question='{question[:60]}...'"
    )

    query_vector = embedder.embed_query(question)

    raw_chunks = vector_store.search(
        query_vector=query_vector,
        workspace_id=workspace_id,
        top_k=top_k,
    )

    if not raw_chunks:
        logger.warning(f"No chunks retrieved for question: '{question[:60]}'")
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

    context, used_chunks = build_context(
        chunks=raw_chunks,
        max_tokens=settings.rag_context_token_budget,
    )

    history = format_history_for_llm(conversation_history)

    llm_result = await generate_rag_response(
        question=question,
        context=context,
        history=history,
    )

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