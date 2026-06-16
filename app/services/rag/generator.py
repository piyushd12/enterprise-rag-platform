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