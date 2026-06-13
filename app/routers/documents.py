import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.dependencies import get_current_user, get_current_workspace
from app.models.document import Document, DocumentStatus
from app.models.user import User
from app.models.workspace import Workspace
from app.schemas.documents import DocumentResponse
from app.services.extractors.factory import SUPPORTED_TYPES
from app.services.storage import StorageService
from app.worker.tasks.ingestion import process_document

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/documents",
    tags=["Documents"],
)

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024 # 50 MB


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for processing",
)
async def upload_document(
    workspace_id: str,
    file: UploadFile = File(..., description="PDF, DOCX, JPEG, or PNG file"),
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    db: AsyncSession = Depends(get_db),
):
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: '{content_type}'. "
                   f"Accepted: PDF, DOCX, JPEG, PNG, TIFF",
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is too large. Maximum size is 50 MB",
        )

    storage = StorageService()
    s3_key = storage.upload_file(
        file_bytes=file_bytes,
        workspace_id=workspace_id,
        filename=file.filename or "untitled",
        content_type=content_type,
    )
    logger.info(f"File uploaded to MinIO: key={s3_key}")

    document = Document(
        workspace_id=workspace_id,
        title=file.filename or "Untitled Document",
        file_name=file.filename or "untitled",
        file_type=content_type,
        file_size=len(file_bytes),
        s3_key=s3_key,
        s3_bucket=settings.s3_bucket_name,
        status=DocumentStatus.PENDING,
    )
    db.add(document)
    await db.flush()

    process_document.delay(document.id)
    logger.info(f"Triggered ingestion task for document_id={document.id}")

    return document


@router.get(
    "",
    response_model=list[DocumentResponse],
    summary="List all documents in this workspace",
)
async def list_documents(
    workspace_id: str,
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document)
        .where(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
    )
    return result.scalars().all()


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a document and its current processing status",
)
async def get_document(
    workspace_id: str,
    document_id: str,
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return document


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its file from storage",
)
async def delete_document(
    workspace_id: str,
    document_id: str,
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        storage = StorageService()
        storage.delete_file(document.s3_key)
    except Exception as e:
        logger.warning(f"Could not delete file from MinIO: {e}")

    await db.delete(document)