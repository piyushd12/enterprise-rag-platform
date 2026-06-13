import logging

from app.worker.celery_app import celery_app
from app.core.database import get_sync_db
from app.models.document import Document, DocumentStatus
from app.services.storage import StorageService
from app.services.extractors.factory import get_extractor

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
    logger.info(f"[Task] Starting ingestion for document_id={document_id}")

    with get_sync_db() as db:
        document = db.get(Document, document_id)
        if not document:
            logger.error(f"[Task] Document {document_id} not found in database")
            return

        try:
            document.status = DocumentStatus.PROCESSING
            db.commit()
            logger.info(f"[Task] Document {document_id} marked as PROCESSING")

            storage = StorageService()
            file_bytes = storage.download_file(document.s3_key)
            logger.info(
                f"[Task] Downloaded {len(file_bytes):,} bytes "
                f"(type={document.file_type})"
            )

            extractor = get_extractor(document.file_type)
            result = extractor.extract(file_bytes)

            if not result["text"].strip():
                raise ValueError(
                    "Extraction produced empty text. "
                    "The file may be a scanned image PDF with no text layer, "
                    "or the file may be corrupted."
                )

            document.extracted_text = result["text"]
            document.page_count = result["page_count"]
            document.status = DocumentStatus.EXTRACTED
            document.error_message = None
            db.commit()

            logger.info(
                f"[Task] Document {document_id} EXTRACTED successfully. "
                f"Pages: {result['page_count']}, "
                f"Characters: {len(result['text']):,}"
            )

        except Exception as exc:
            logger.error(f"[Task] Ingestion failed for {document_id}: {exc}")
            document.status = DocumentStatus.FAILED
            document.error_message = str(exc)[:2000]
            db.commit()

            raise