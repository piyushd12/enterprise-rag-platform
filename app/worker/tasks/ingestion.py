import logging

from app.worker.celery_app import celery_app
from app.core.database import get_sync_db
from app.models.document import Document, DocumentStatus
from app.models.chunk import DocumentChunk
from app.services.storage import StorageService
from app.services.extractors.factory import get_extractor
from app.services.chunker import chunk_document
from app.services.embedder import embedder
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    name="ingestion.process_document",
)
def process_document(self, document_id: str):
    logger.info(f"[Task] Starting full ingestion for document_id={document_id}")

    with get_sync_db() as db:
        document = db.get(Document, document_id)
        if not document:
            logger.error(f"[Task] Document {document_id} not found — skipping")
            return

        try:
            document.status = DocumentStatus.PROCESSING
            db.commit()

            storage = StorageService()
            file_bytes = storage.download_file(document.s3_key)
            logger.info(f"[Task] Downloaded {len(file_bytes):,} bytes")

            extractor = get_extractor(document.file_type)
            extraction_result = extractor.extract(file_bytes)

            if not extraction_result["text"].strip():
                raise ValueError(
                    "Extraction produced empty text. "
                    "The file may be scanned or corrupted."
                )

            extracted_text = extraction_result["text"]
            page_count = extraction_result["page_count"]
            page_boundaries = extraction_result.get("page_boundaries", [0])

            document.extracted_text = extracted_text
            document.page_count = page_count
            document.status = DocumentStatus.EXTRACTED
            db.commit()
            logger.info(
                f"[Task] Extracted {len(extracted_text):,} chars, "
                f"{page_count} pages"
            )

            chunks = chunk_document(
                text=extracted_text,
                document_id=document_id,
                workspace_id=document.workspace_id,
                page_boundaries=page_boundaries,
                chunk_size=400,
                chunk_overlap=50,
            )

            if not chunks:
                raise ValueError(
                    "Chunking produced zero chunks. "
                    "The extracted text may be too short or contain only noise."
                )
            logger.info(f"[Task] Produced {len(chunks)} chunks")

            chunk_texts = [c["chunk_text"] for c in chunks]
            embeddings = embedder.embed_texts(chunk_texts, batch_size=32)
            logger.info(f"[Task] Embedded {len(embeddings)} chunks")

            vector_store.delete_document_vectors(
                document_id=document_id,
                workspace_id=document.workspace_id,
            )
            point_ids = vector_store.upsert_chunks(chunks, embeddings)
            logger.info(f"[Task] Upserted {len(point_ids)} vectors to Qdrant")

            existing_chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).all()
            for c in existing_chunks:
                db.delete(c)
            db.flush()

            for chunk, point_id in zip(chunks, point_ids):
                db_chunk = DocumentChunk(
                    id=chunk["id"],
                    document_id=chunk["document_id"],
                    workspace_id=chunk["workspace_id"],
                    chunk_index=chunk["chunk_index"],
                    chunk_text=chunk["chunk_text"],
                    page_num=chunk["page_num"],
                    char_start=chunk["char_start"],
                    char_end=chunk["char_end"],
                    token_count=chunk["token_count"],
                    qdrant_point_id=str(point_id),
                )
                db.add(db_chunk)

            document.status = DocumentStatus.CHUNKED
            document.error_message = None
            db.commit()

            logger.info(
                f"[Task] Document {document_id} fully processed. "
                f"Status: CHUNKED. Chunks: {len(chunks)}"
            )

        except Exception as exc:
            logger.error(f"[Task] Ingestion failed for {document_id}: {exc}")
            document.status = DocumentStatus.FAILED
            document.error_message = str(exc)[:2000]
            db.commit()
            raise