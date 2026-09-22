"""
Chat session endpoints — list saved chats and load one's full history,
powering the sidebar in the UI. Chats are created implicitly by
POST /chat/stream (see routes/chat.py) rather than through this router.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from rag_app.api.dependencies import get_chat_store
from rag_app.api.schemas import ChatDetailResponse, ChatListResponse, ChatSummary
from rag_app.core.interfaces import ChatStore

router = APIRouter(tags=["chats"])


@router.get("/chats", response_model=ChatListResponse)
async def list_chats(chat_store: ChatStore = Depends(get_chat_store)) -> ChatListResponse:
    """List every saved chat, most recently active first."""
    chats = await chat_store.list_chats()
    return ChatListResponse(chats=[ChatSummary(**c) for c in chats])


@router.get("/chats/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(chat_id: str, chat_store: ChatStore = Depends(get_chat_store)) -> ChatDetailResponse:
    """Get one chat's full message history."""
    chat = await chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return ChatDetailResponse(**chat)
