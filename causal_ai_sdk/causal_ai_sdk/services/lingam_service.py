"""LiNGAM Causal Discovery service."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from causal_ai_sdk.contracts.requests import CDLingamRunRequestContract
from causal_ai_sdk.exceptions import ValidationError
from causal_ai_sdk.models.cd import UploadedData
from causal_ai_sdk.services.base_cd_service import BaseCDService


class LingamService(BaseCDService):
    """LiNGAM Causal Discovery service.

    This service provides methods for uploading a single dataset and
    running LayeredLiNGAM causal discovery.
    """

    def _validate_upload_requirements(self, file_paths: List[Path]) -> Dict[Path, Any]:
        """Validate LiNGAM upload requirements (single file).

        Args:
            file_paths (List[Path]): List of file paths to validate

        Returns:
            Dict[Path, Any]: Empty cache (LiNGAM does not pre-read files).

        Raises:
            ValidationError: If not exactly one file or format invalid
        """
        if len(file_paths) > 1:
            raise ValidationError(
                message="LiNGAM only accepts a single file, not multiple files",
                status_code=None,
                response_data=None,
            )

        if len(file_paths) != 1:
            raise ValidationError(
                message="One file path is required",
                status_code=None,
                response_data=None,
            )

        for fp in file_paths:
            if not fp.exists():
                raise ValidationError(
                    message=f"File not found: {fp}",
                    status_code=None,
                    response_data=None,
                )
            self._validate_file_format(fp)

        return {}

    def _build_upload_body(self, file_paths: List[Path]) -> tuple[bytes, str]:
        """Build upload body: single file as raw bytes.

        Args:
            file_paths (List[Path]): Exactly one file path (already validated).

        Returns:
            tuple[bytes, str]: (file bytes, Content-Type)
        """
        fp = file_paths[0]
        body = fp.read_bytes()
        content_type = self._content_type_for_path(fp)
        return body, content_type

    async def upload_data_for_lingam(
        self,
        session_uuid: str,
        file_path: Union[str, Path],
    ) -> UploadedData:
        """Upload single dataset for LiNGAM causal discovery.

        This method automatically:
        1. Validates that exactly one file is provided
        2. Validates that the file is in a supported format (JSON, CSV)
        3. Gets a presigned upload URL
        4. Uploads the file as raw bytes to S3

        Args:
            session_uuid (str): Session UUID
            file_path (Union[str, Path]): Path to a single file (CSV or JSON)

        Returns:
            UploadedData: Uploaded data reference (task_id, session_uuid, s3_key).
        """
        file_path_list = [Path(file_path)]
        upload_url, _ = await self._upload_data_internal(session_uuid, file_path_list)
        return UploadedData(
            task_id=upload_url.task_id,
            session_uuid=session_uuid,
            s3_key=upload_url.s3_key,
        )

    async def run_lingam(
        self,
        uploaded_data: UploadedData,
        threshold: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit a LayeredLiNGAM causal discovery task.

        Args:
            uploaded_data (UploadedData): From upload_data_for_lingam.
            threshold (Optional[float]): Significance threshold (default: None).
                Use None to omit and let backend defaults apply; if set, must be in (0, 1].
            params (Optional[Dict[str, Any]]): Optional algorithm params (e.g. gamma, delta)

        Returns:
            Dict: Task response (status: queued)
        """
        ref = uploaded_data
        path = self._contract_path("cd.run_lingam", uuid=ref.session_uuid)
        payload = CDLingamRunRequestContract(
            task_id=ref.task_id,
            s3_key=ref.s3_key,
            threshold=threshold,
            params=params,
        )
        json_data = payload.model_dump(exclude_none=True)
        response = await self._make_request("POST", path, json_data=json_data)
        return response
