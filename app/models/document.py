import enum

from sqlalchemy import Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, generate_uuid


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"        # uploaded, waiting in queue
    PROCESSING = "processing"  # Celery worker is actively processing it
    EXTRACTED = "extracted"    # text extracted successfully
    CHUNKED = "chunked"        # chunked and embedded
    FAILED = "failed"          # extraction failed


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    s3_key: Mapped[str] = mapped_column(String(1000), nullable=False)
    s3_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # can be very long
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    workspace: Mapped["Workspace"] = relationship(back_populates="documents")

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, title={self.title!r}, status={self.status!r})"