"""
Compatibility shim for a ragas <-> langchain-community version mismatch.

ragas 0.2.x eagerly imports langchain_community.chat_models.vertexai at
module load time, even though this project never uses Vertex AI (Groq
and OpenRouter only). Recent langchain-community releases (the 0.4.x
"sunset" line, where individual integrations are being removed) dropped
that module entirely, so a bare `import ragas` fails with
ModuleNotFoundError unless something registers a stand-in first.

Call apply() before importing ragas anywhere (see evaluation/__init__.py).
"""

from __future__ import annotations

import sys
import types

_SHIMMED_MODULE = "langchain_community.chat_models.vertexai"


def apply() -> None:
    """Register a stub for the missing module if it isn't already importable."""
    if _SHIMMED_MODULE in sys.modules:
        return

    try:
        import langchain_community.chat_models.vertexai  # noqa: F401

        return  # already importable — no shim needed
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType(_SHIMMED_MODULE)

    class ChatVertexAI:  # pragma: no cover - stub is never actually invoked
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "ChatVertexAI is a compatibility stub — Vertex AI is not "
                "supported by this project (Groq/OpenRouter only)."
            )

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[_SHIMMED_MODULE] = shim
