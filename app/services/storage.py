import uuid
import logging
from botocore.exceptions import ClientError
import boto3

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name="us-east-1",
        )
        self.bucket = settings.s3_bucket_name

    def ensure_bucket_exists(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            logger.info(f"S3 bucket '{self.bucket}' already exists")
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)
            logger.info(f"Created S3 bucket '{self.bucket}'")

    def upload_file(
        self,
        file_bytes: bytes,
        workspace_id: str,
        filename: str,
        content_type: str,
    ) -> str:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
        key = f"workspaces/{workspace_id}/documents/{uuid.uuid4()}.{ext}"

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=file_bytes,
            ContentType=content_type,
        )
        logger.info(f"Uploaded {len(file_bytes)} bytes to s3://{self.bucket}/{key}")
        return key

    def download_file(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        file_bytes = response["Body"].read()
        logger.info(f"Downloaded {len(file_bytes)} bytes from s3://{self.bucket}/{key}")
        return file_bytes

    def delete_file(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)
        logger.info(f"Deleted s3://{self.bucket}/{key}")