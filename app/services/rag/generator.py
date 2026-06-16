import logging
import litellm

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

RAG_SYSTEM_PROMPT = """
You are a precise document assistant. Your only job is to answer questions based on the document context provided to you.

STRICT RULES:
1. Answer ONLY using information found in the provided context sections below.
2. If the context does not contain enough information to answer, respond with exactly:
   "The provided documents don't contain enough information to answer this question."
3. Cite your sources using the source labels provided, like this: (Source 1) or (Source 2, Page 3).
4. Do NOT add any information from your training that is not present in the context.
5. Be concise. Aim for clarity over length.
"""


def _build_model_string() -> str:
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return f"ollama/{settings.ollama_model}"
    elif provider == "openai":
        return "gpt-4o"
    elif provider == "groq":
        return "groq/llama-3.1-70b-versatile"
    elif provider == "bedrock":
        return "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. "
                         f"Valid options: ollama, openai, groq, bedrock")


def _build_messages(
    question: str,
    context: str,
    history: list[dict],
) -> list[dict]:
    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

    if history:
        messages.extend(history)

    user_content = f"""DOCUMENT CONTEXT:
{context}


QUESTION: {question}"""

    messages.append({"role": "user", "content": user_content})
    return messages


async def generate_rag_response(
    question: str,
    context: str,
    history: list[dict],
) -> dict:
    model = _build_model_string()
    messages = _build_messages(question, context, history)

    logger.info(f"Calling LLM: model={model}, messages={len(messages)}")

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=settings.max_response_tokens,
            temperature=0.1,
            api_base=settings.ollama_base_url if settings.llm_provider == "ollama" else None,
        )

        answer = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else None

        logger.info(
            f"LLM response received: {len(answer)} chars, "
            f"tokens_used={tokens_used}"
        )

        return {
            "answer": answer,
            "tokens_used": tokens_used,
            "model": model,
        }

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise RuntimeError(f"LLM generation failed: {str(e)}") from e
    

# app/services/rag/generator.py
# DELETE the _CONTEXT_SIGNALS set completely — remove those lines entirely

async def rewrite_query_for_retrieval(
    question: str,
    history: list[dict],
) -> str:
    """
    Uses the LLM to read the actual conversation history and decide
    whether the question needs rewriting for standalone retrieval.

    The LLM returns the question unchanged if it's already standalone.
    The LLM rewrites it if it references something from the conversation.

    No keyword matching. No hardcoded signals. The LLM reads the real
    conversation and makes the judgment itself.
    """
    if not history:
        return question

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history[-6:]
    )

    prompt = (
        "You are a search query optimizer.\n\n"
        "Read the conversation history below and the follow-up question.\n"
        "Your task: rewrite the follow-up question as a fully standalone "
        "search query that can be understood with zero context from the conversation.\n\n"
        "Rules:\n"
        "- If the question already makes complete sense on its own, return it EXACTLY as written.\n"
        "- If the question references anything from the conversation "
        "(a person, topic, document, decision, or uses words like 'that', 'it', 'him', "
        "'this', 'tell me more', 'the first one', 'what you just said', etc.), "
        "rewrite it so those references are replaced with the actual content they refer to.\n"
        "- Return ONLY the rewritten question. No explanation. No preamble. No quotes.\n\n"
        f"CONVERSATION HISTORY:\n{history_text}\n\n"
        f"FOLLOW-UP QUESTION: {question}\n\n"
        "STANDALONE SEARCH QUERY:"
    )

    try:
        response = await litellm.acompletion(
            model=_build_model_string(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
            api_base=(
                settings.ollama_base_url
                if settings.llm_provider == "ollama"
                else None
            ),
        )
        rewritten = response.choices[0].message.content.strip()

        if not rewritten or len(rewritten) > 400:
            return question

        logger.info(
            f"Query rewrite: '{question}' → '{rewritten}'"
            if rewritten.lower() != question.lower()
            else f"Query unchanged: '{question}'"
        )
        return rewritten

    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")
        return question