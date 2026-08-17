"""
Individual node functions for the RAG LangGraph pipeline.

Each node is a pure function: takes RAGState, returns a partial state update.
Nodes are independently testable and traced by LangSmith as child runs.
"""

from __future__ import annotations

from rag_app.core.interfaces import EmbeddingProvider, LLMProvider, VectorStore
from rag_app.rag.state import RAGState


def make_retrieve_node(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
):
    """Factory for the retrieve node — closes over providers."""

    async def retrieve(state: RAGState) -> dict:
        """Embed the query and search the vector store."""
        query = state["query"]
        top_k = state.get("top_k", 5)
        filters = state.get("filters")

        # Embed query
        query_embedding = embedding_provider.embed_query(query)

        # Search vector store
        chunks = await vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        return {
            "query_embedding": query_embedding,
            "retrieved_chunks": chunks,
            # Pass-through for rerank (Phase 3 will replace this)
            "reranked_chunks": chunks,
        }

    return retrieve


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
