"""Causal Discovery upload / task reference models."""

from pydantic import BaseModel, Field


class CDUploadURL(BaseModel):
    """Upload URL model for Causal Discovery data upload.

    Attributes:
        uuid (str): Session UUID
        task_id (str): Task ID for the CD task
        s3_key (str): S3 key for the uploaded file (internal use)
        upload_url (str): Presigned URL for uploading to S3
        expires_in (int): Expiration time in seconds
    """

    uuid: str
    task_id: str
    s3_key: str
    upload_url: str
    expires_in: int


class UploadedData(BaseModel):
    """Model for uploaded data reference (hides s3_key from client).

    Attributes:
        task_id (str): Task ID for the CD task
        session_uuid (str): Session UUID
        s3_key (str): S3 key (internal use only)
    """

    task_id: str
    session_uuid: str
    s3_key: str = Field(description="Internal use only")

    @property
    def _s3_key(self) -> str:
        """Alias for s3_key for backward compatibility.

        Returns:
            The S3 key string.
        """
        return self.s3_key
