import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.dependencies import get_current_user, get_current_workspace
from app.models.conversation import Conversation, Message, MessageRole
from app.models.user import User
from app.models.workspace import Workspace
from app.schemas.chat import (
    ChatRequest, ChatResponse, ConversationResponse,
    MessageResponse, SourceCitation
)
from app.services.rag.pipeline import run_rag_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}",
    tags=["Chat"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message and get an AI answer from your documents",
)
async def chat(
    workspace_id: str,
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    db: AsyncSession = Depends(get_db),
):
    if request.conversation_id:
        result = await db.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(
                Conversation.id == request.conversation_id,
                Conversation.workspace_id == workspace_id,
                Conversation.user_id == current_user.id,
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
    else:
        title = request.query[:100] + ("..." if len(request.query) > 100 else "")
        conversation = Conversation(
            workspace_id=workspace_id,
            user_id=current_user.id,
            title=title,
        )
        db.add(conversation)
        await db.flush()
        # conversation.messages = []

    try:
        conversation_history = conversation.messages if request.conversation_id else []
        rag_result = await run_rag_pipeline(
            question=request.query,
            workspace_id=workspace_id,
            conversation_history=conversation_history,
        )
    except RuntimeError as e:
        logger.error(f"RAG pipeline failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI service temporarily unavailable: {str(e)}",
        )

    user_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content=request.query,
        sources=None,
    )
    db.add(user_message)

    assistant_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content=rag_result["answer"],
        sources=rag_result["sources"],
        tokens_used=rag_result["tokens_used"],
    )
    db.add(assistant_message)
    await db.flush()

    sources = [SourceCitation(**s) for s in rag_result["sources"]]

    logger.info(
        f"Chat response: conversation={conversation.id}, "
        f"sources={len(sources)}, tokens={rag_result['tokens_used']}"
    )

    return ChatResponse(
        conversation_id=conversation.id,
        message_id=assistant_message.id,
        answer=rag_result["answer"],
        sources=sources,
        chunks_retrieved=rag_result["chunks_retrieved"],
        chunks_used=rag_result["chunks_used"],
        tokens_used=rag_result["tokens_used"],
    )


@router.get(
    "/conversations",
    response_model=list[ConversationResponse],
    summary="List all conversations in this workspace",
)
async def list_conversations(
    workspace_id: str,
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.workspace_id == workspace_id,
            Conversation.user_id == current_user.id,
        )
        .order_by(Conversation.created_at.desc())
    )
    return result.scalars().all()


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
    summary="Get all messages in a conversation",
)
async def get_conversation_messages(
    workspace_id: str,
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(
            Conversation.id == conversation_id,
            Conversation.workspace_id == workspace_id,
            Conversation.user_id == current_user.id,
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return [
        MessageResponse(
            id=msg.id,
            role=msg.role.value,
            content=msg.content,
            sources=msg.sources,
        )
        for msg in conversation.messages
    ]