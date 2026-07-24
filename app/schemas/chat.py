from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = Field(
        default=None,
        description="Pass an existing conversation_id to continue a conversation. "
                    "Omit or pass null to start a new conversation."
    )


class SourceCitation(BaseModel):
    document_id: str
    chunk_text: str
    page_num: int
    score: float

    model_config = {"from_attributes": True}


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    sources: list[SourceCitation]
    chunks_retrieved: int
    chunks_used: int
    tokens_used: int | None = None


class ConversationResponse(BaseModel):
    id: str
    title: str | None
    workspace_id: str

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: list | None = None

    model_config = {"from_attributes": True}