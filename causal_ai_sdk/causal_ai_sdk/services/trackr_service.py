"""TraCKR Causal Discovery service."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from causal_ai_sdk.contracts.requests import (
    CDMatchSetRequestContract,
    CDTrackrMatchStartRequestContract,
    CDTrackrRunRequestContract,
)
from causal_ai_sdk.exceptions import APIError, ValidationError
from causal_ai_sdk.models.cd import UploadedData
from causal_ai_sdk.services.base_cd_service import BaseCDService
from causal_ai_sdk.utils.polling import poll_until_ready_or_fail
from pydantic import ValidationError as PydanticValidationError


class TraCKRService(BaseCDService):
    """TraCKR Causal Discovery service.

    This service provides methods for uploading single dataset,
    managing column matching, and executing TraCKR causal discovery tasks.
    """

    def _validate_upload_requirements(self, file_paths: List[Path]) -> Dict[Path, Any]:
        """Validate TraCKR upload requirements.

        Args:
            file_paths (List[Path]): List of file paths to validate

        Returns:
            Dict[Path, Any]: Empty cache (TraCKR does not pre-read files).

        Raises:
            ValidationError: If validation fails (TraCKR only accepts single file)
        """
        if len(file_paths) > 1:
            raise ValidationError(
                message="TraCKR only accepts a single file, not multiple files",
                status_code=None,
                response_data=None,
            )

        if len(file_paths) == 0:
            raise ValidationError(
                message="At least one file path is required",
                status_code=None,
                response_data=None,
            )

        # Validate file existence then format
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

    async def upload_data_for_trackr(
        self,
        session_uuid: str,
        file_path: Union[str, Path],
    ) -> UploadedData:
        """Upload single dataset for TraCKR causal discovery.

        This method automatically:
        1. Validates that exactly one file is provided
        2. Validates that the file is in a supported format
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

    async def wait_for_matching(
        self,
        session_uuid: str,
        timeout: int = 300,
        interval: int = 3,
        matching_task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Wait for TraCKR matching computation to complete (with automatic polling).

        Args:
            session_uuid (str): Session UUID
            timeout (int): Maximum time to wait in seconds (default: 300)
            interval (int): Time between polls in seconds (default: 3)
            matching_task_id (Optional[str]): Matching task ID from start_trackr_matching
                (required by API)

        Returns:
            Dict: Final matching result
        """

        async def check() -> Dict[str, Any]:
            return await self.get_trackr_matching(session_uuid, matching_task_id)

        def on_failed(result: Dict[str, Any]) -> None:
            error_msg = result.get("error") or "Unknown error"
            raise ValidationError(
                message=f"Matching failed: {error_msg}",
                status_code=None,
                response_data=None,
            )

        return await poll_until_ready_or_fail(
            check_func=check,
            is_ready=lambda r: r.get("status") == "completed",
            is_failed=lambda r: r.get("status") == "failed",
            on_failed=on_failed,
            timeout=timeout,
            interval=interval,
            timeout_error_message=f"Matching did not complete within timeout ({timeout}s)",
            retry_exceptions=(APIError,),
            retry_if=lambda e: getattr(e, "status_code", None) in (500, 503),
        )

    async def start_trackr_matching(
        self,
        uploaded_data: UploadedData,
        source_kg_id: str,
        target_metadata: Optional[Dict] = None,
        source_metadata: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Start TraCKR column matching computation.

        Args:
            uploaded_data (UploadedData): From upload_data_for_trackr.
            source_kg_id (str): Source knowledge graph ID (must exist in KG service)
            target_metadata (Optional[Dict]): Optional column descriptions for target dataset.
            source_metadata (Optional[Dict]): Optional column descriptions for source KG.
            params (Optional[Dict]): Optional matching parameters

        Returns:
            Dict: Matching task response (status: pending)
        """
        ref = uploaded_data
        path = self._contract_path("cd.start_trackr_matching", uuid=ref.session_uuid)
        payload = CDTrackrMatchStartRequestContract(
            target_s3_key=ref.s3_key,
            source_kg_id=source_kg_id,
            target_metadata=target_metadata,
            source_metadata=source_metadata,
            params=params,
        )
        json_data: Dict[str, Union[str, Dict]] = payload.model_dump(exclude_none=True)
        response = await self._make_request("POST", path, json_data=json_data)
        return response

    async def get_trackr_matching(
        self, session_uuid: str, matching_task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get TraCKR column matching results.

        Args:
            session_uuid (str): Session UUID
            matching_task_id (Optional[str]): Matching task ID from start_trackr_matching
                (required by API)

        Returns:
            Dict: Matching result with current_matched, knowledge_coverage, etc.
        """
        path = self._contract_path("cd.get_trackr_matching", uuid=session_uuid)
        params = {"matching_task_id": matching_task_id} if matching_task_id else None
        response = await self._make_request("GET", path, params=params)
        return response

    async def set_trackr_matching(
        self,
        session_uuid: str,
        matching: Dict[str, str],
        matching_task_id: Optional[str] = None,
    ) -> None:
        """Set/update TraCKR column matching results (update of existing state).

        Pass matching_task_id from start_trackr_matching to update that matching.
        The API requires matching to be a non-empty dict.

        Args:
            session_uuid (str): Session UUID
            matching (Dict[str, str]): Column mapping (target column -> source column).
                Must be non-empty. Use "" for target columns with no match.
            matching_task_id (Optional[str]): Matching task ID from start_trackr_matching
                (required to update existing)

        Raises:
            ValidationError: If matching is empty (400).
        """
        if not matching:
            raise ValidationError(
                message="'matching' must be a non-empty dictionary",
                status_code=400,
                response_data=None,
            )
        path = self._contract_path("cd.set_trackr_matching", uuid=session_uuid)
        payload = CDMatchSetRequestContract(matching=matching, matching_task_id=matching_task_id)
        json_data = payload.model_dump(exclude_none=True)
        await self._make_request("POST", path, json_data=json_data)

    async def run_trackr(
        self,
        uploaded_data: UploadedData,
        transferred_knowledge: Dict[str, str],
        matching_task_id: str,
        threshold: Optional[float] = 0.01,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Submit a TraCKR causal discovery task with knowledge transfer.

        Args:
            uploaded_data (UploadedData): From upload_data_for_trackr.
            transferred_knowledge (Dict[str, str]): Knowledge transfer metadata (required).
                Must contain both "session_uuid" and "kg_id".
            matching_task_id (str): Matching task ID from start_trackr_matching (required).
            threshold (Optional[float]): Significance threshold (default: 0.01).
                Use ``None`` to omit threshold and let backend defaults apply.
            params (Optional[Dict]): Optional TraCKR parameters

        Returns:
            Dict: Task response (status: queued)

        Raises:
            ValidationError: If the request payload does not align with the contract (400).
            NotFoundError: If the knowledge graph does not exist (404).  # noqa: DAR402
            APIError: If KG has no S3 reference or validation fails (500).  # noqa: DAR402
        """
        ref = uploaded_data
        try:
            payload = CDTrackrRunRequestContract(
                task_id=ref.task_id,
                s3_key=ref.s3_key,
                transferred_knowledge=transferred_knowledge,
                threshold=threshold,
                params=params,
                matching_task_id=matching_task_id,
            )
        except PydanticValidationError as e:
            raise ValidationError(
                message=str(e),
                status_code=400,
                response_data=None,
            ) from e

        path = self._contract_path("cd.run_trackr", uuid=ref.session_uuid)
        json_data: Dict[str, Union[str, float, Dict]] = payload.model_dump(exclude_none=True)
        response = await self._make_request("POST", path, json_data=json_data)
        return response

    async def delete_trackr_matching(
        self, session_uuid: str, matching_task_id: Optional[str] = None
    ) -> None:
        """Delete TraCKR matching state and S3 artifacts for the session.

        Corresponds to DELETE /cd/match/{session_uuid}
        ?matching_type=trackr&matching_task_id=...
        Use after run_trackr to clean up matching state.

        Args:
            session_uuid (str): Session UUID
            matching_task_id (Optional[str]): Matching task ID from start_trackr_matching
                (required by API)

        Raises:
            NotFoundError: If matching state does not exist (404).  # noqa: DAR402
            APIError: If matching is still pending (409).  # noqa: DAR402
        """
        await self.delete_matching(session_uuid, "trackr", matching_task_id)
