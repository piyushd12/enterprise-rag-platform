import logging
from app.services.chunker import count_tokens

logger = logging.getLogger(__name__)


def build_context(
    chunks: list[dict],
    max_tokens: int = 3000,
) -> tuple[str, list[dict]]:
    context_parts = []
    used_chunks = []
    total_tokens = 0

    for i, chunk in enumerate(chunks):
        chunk_text = chunk["chunk_text"]

        source_label = f"[Source {i + 1} | Page {chunk['page_num']}]"
        formatted = f"{source_label}\n{chunk_text}"
        chunk_tokens = count_tokens(formatted)

        if total_tokens + chunk_tokens > max_tokens:
            logger.debug(
                f"Context budget reached at chunk {i}. "
                f"Used {total_tokens}/{max_tokens} tokens."
            )
            break

        context_parts.append(formatted)
        used_chunks.append(chunk)
        total_tokens += chunk_tokens

    if not used_chunks:
        logger.warning("Context builder produced empty context — no chunks fit the budget")

    context_string = "\n\n---\n\n".join(context_parts)

    logger.info(
        f"Context built: {len(used_chunks)}/{len(chunks)} chunks used, "
        f"{total_tokens} tokens"
    )
    return context_string, used_chunks


def format_history_for_llm(messages: list) -> list[dict]:
    history = []
    for msg in messages:
        history.append({
            "role": msg.role.value,
            "content": msg.content,
        })
    return history[-6:]