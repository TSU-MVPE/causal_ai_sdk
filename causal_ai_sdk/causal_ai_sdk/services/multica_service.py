"""MultiCa Causal Discovery service."""

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from causal_ai_sdk.contracts.requests import (
    CDMatchSetRequestContract,
    CDMulticaMatchStartRequestContract,
    CDMulticaRunRequestContract,
)
from causal_ai_sdk.exceptions import APIError, ValidationError
from causal_ai_sdk.models.cd import UploadedData
from causal_ai_sdk.services.base_cd_service import BaseCDService
from causal_ai_sdk.utils.dataset_schema import validate_columns_data
from causal_ai_sdk.utils.polling import poll_until_ready_or_fail
from pydantic import ValidationError as PydanticValidationError

logger = logging.getLogger(__name__)

# Log matching progress every this many seconds (avoids spamming when polling every 3s)
_MATCHING_PROGRESS_LOG_INTERVAL = 30


class MultiCaService(BaseCDService):
    """MultiCa Causal Discovery service.

    This service provides methods for uploading multiple datasets,
    managing column matching, and executing MultiCa causal discovery tasks.
    """

    def _validate_multica_files(self, file_paths: List[Path]) -> Dict[Path, Any]:
        """Validate MultiCa file requirements and return cached parsed data when applicable.

        When a single JSON file is validated, it is read once and returned in the cache
        so the upload step does not read it again. Requirements:
        - JSON files: Can be used alone if they contain multiple datasets (in "data" array)
        - CSV files: Require at least 2 files
        - All files must be in supported formats (JSON, CSV)

        Args:
            file_paths (List[Path]): List of file paths to validate

        Returns:
            Dict[Path, Any]: Cache of path -> parsed data for files already read (e.g. single JSON).

        Raises:
            ValidationError: If validation fails
        """
        supported_extensions = {".json", ".csv"}
        read_cache: Dict[Path, Any] = {}

        # Validate file existence and format
        for fp in file_paths:
            if not fp.exists():
                raise ValidationError(
                    message=f"File not found: {fp}",
                    status_code=None,
                    response_data=None,
                )
            if fp.suffix.lower() not in supported_extensions:
                raise ValidationError(
                    message=(
                        f"Unsupported file format: {fp.suffix}. " "Supported formats: JSON, CSV"
                    ),
                    status_code=None,
                    response_data=None,
                )

        # At least 2 files required (or single JSON with multiple datasets)
        if len(file_paths) >= 2:
            return read_cache
        if len(file_paths) == 0:
            raise ValidationError(
                message="MultiCa requires at least 2 files, found 0",
                status_code=None,
                response_data=None,
            )

        single_file = file_paths[0]
        if single_file.suffix.lower() != ".json":
            raise ValidationError(
                message=(
                    f"MultiCa requires at least 2 {single_file.suffix.upper()} files, "
                    "found 1. JSON files can be used alone if they contain "
                    "multiple datasets."
                ),
                status_code=None,
                response_data=None,
            )

        # Single JSON file: read once for validation and cache for upload step
        try:
            with open(single_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValidationError(
                message=f"Invalid JSON file: {e}",
                status_code=None,
                response_data=None,
            ) from e
        except Exception as e:
            raise ValidationError(
                message=f"Failed to read JSON file: {e}",
                status_code=None,
                response_data=None,
            ) from e

        if not (isinstance(data, dict) and "data" in data):
            raise ValidationError(
                message=(
                    "JSON file must contain 'data' key with array format, "
                    "or provide at least 2 files"
                ),
                status_code=None,
                response_data=None,
            )
        if not (isinstance(data["data"], list) and len(data["data"]) >= 2):
            raise ValidationError(
                message=(
                    f"JSON file must contain at least 2 datasets "
                    f"(in 'data' array), found {len(data.get('data', []))}"
                ),
                status_code=None,
                response_data=None,
            )

        # Validate each dataset has columns + data schema
        for i, dataset in enumerate(data["data"]):
            validate_columns_data(dataset, context=f"JSON dataset at index {i}")

        read_cache[single_file] = data
        return read_cache

    def _validate_upload_requirements(self, file_paths: List[Path]) -> Dict[Path, Any]:
        """Validate MultiCa upload requirements; return cached parsed data for single JSON.

        Args:
            file_paths (List[Path]): List of file paths to validate

        Returns:
            Dict[Path, Any]: Cache of path -> parsed data when already read during validation.
        """
        return self._validate_multica_files(file_paths)

    def _build_upload_body(self, file_paths: List[Path]) -> Tuple[bytes, str]:
        """Build upload body: single file as raw bytes, or multiple files as ZIP.

        Args:
            file_paths (List[Path]): File paths (already validated).

        Returns:
            Tuple[bytes, str]: (body bytes, Content-Type)
        """
        if len(file_paths) == 1:
            body = file_paths[0].read_bytes()
            content_type = self._content_type_for_path(file_paths[0])
            return body, content_type
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in file_paths:
                zf.write(fp, fp.name)
        return buf.getvalue(), "application/zip"

    async def upload_data_for_multica(
        self,
        session_uuid: str,
        file_paths: Union[str, Path, List[Union[str, Path]]],
    ) -> UploadedData:
        """Upload multiple datasets for MultiCa causal discovery.

        This method automatically:
        1. Validates that at least 2 files are provided
        2. Validates that all files are in supported formats
        3. Gets a presigned upload URL
        4. Uploads raw file bytes (single JSON) or a ZIP of files to S3

        Args:
            session_uuid (str): Session UUID
            file_paths (Union[str, Path, List[Union[str, Path]]]):
                Single file path or list of file paths (CSV or JSON)

        Returns:
            UploadedData: Uploaded data reference (task_id, session_uuid, s3_key).
        """
        # Normalize input to list
        if isinstance(file_paths, (str, Path)):
            file_path_list = [Path(file_paths)]
        else:
            file_path_list = [Path(fp) for fp in file_paths]

        # Validate before conversion
        self._validate_multica_files(file_path_list)

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
        retry_on_5xx: bool = True,
    ) -> Dict[str, Any]:
        """Wait for MultiCa matching computation to complete (with automatic polling).

        Args:
            session_uuid (str): Session UUID
            timeout (int): Maximum time to wait in seconds (default: 300)
            interval (int): Time between polls in seconds (default: 3)
            matching_task_id (Optional[str]): Matching task ID from start_multica_matching
                (required by API)
            retry_on_5xx (bool): If True, retry on 500/503 until timeout; if False, raise
                on first 5xx so callers can see the API error body immediately (default: True).

        Returns:
            Dict: Final matching result
        """

        async def check() -> Dict[str, Any]:
            return await self.get_multica_matching(session_uuid, matching_task_id)

        def on_failed(result: Dict[str, Any]) -> None:
            error_msg = result.get("error") or "Unknown error"
            raise ValidationError(
                message=f"Matching failed: {error_msg}",
                status_code=None,
                response_data=None,
            )

        last_log = [0.0]

        def on_poll(elapsed: float, state: Dict[str, Any]) -> None:
            if elapsed - last_log[0] >= _MATCHING_PROGRESS_LOG_INTERVAL:
                logger.info(
                    "MultiCa matching in progress (%.0fs elapsed, status=%s)...",
                    elapsed,
                    state.get("status", "?"),
                )
                last_log[0] = elapsed

        return await poll_until_ready_or_fail(
            check_func=check,
            is_ready=lambda r: r.get("status") == "completed",
            is_failed=lambda r: r.get("status") == "failed",
            on_failed=on_failed,
            timeout=timeout,
            interval=interval,
            timeout_error_message=f"Matching did not complete within timeout ({timeout}s)",
            retry_exceptions=(APIError,),
            retry_if=(
                (lambda e: getattr(e, "status_code", None) in (500, 503))
                if retry_on_5xx
                else (lambda e: False)
            ),
            on_poll=on_poll,
        )

    async def start_multica_matching(
        self,
        uploaded_data: UploadedData,
        metadata: Optional[List[Dict]] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Start MultiCa column matching computation.

        Args:
            uploaded_data (UploadedData): From upload_data_for_multica.
            metadata (Optional[List[Dict]]): Optional metadata for each dataset
            params (Optional[Dict]): Optional matching parameters (top_n, gamma, q_threshold)

        Returns:
            Dict: Matching task response (status: pending)
        """
        ref = uploaded_data
        path = self._contract_path("cd.start_multica_matching", uuid=ref.session_uuid)
        payload = CDMulticaMatchStartRequestContract(
            s3_key=ref.s3_key,
            metadata=metadata,
            params=params,
        )
        json_data: Dict[str, Union[str, List[Dict], Dict]] = payload.model_dump(exclude_none=True)
        response = await self._make_request("POST", path, json_data=json_data)
        return response

    async def get_multica_matching(
        self, session_uuid: str, matching_task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get MultiCa column matching results.

        Args:
            session_uuid (str): Session UUID
            matching_task_id (Optional[str]): Matching task ID from start_multica_matching
                (required by API)

        Returns:
            Dict: Matching result with current_matched, match_rate, etc.
        """
        path = self._contract_path("cd.get_multica_matching", uuid=session_uuid)
        params = {"matching_task_id": matching_task_id} if matching_task_id else None
        response = await self._make_request("GET", path, params=params)
        return response

    async def set_multica_matching(
        self,
        session_uuid: str,
        matching: Dict[str, str],
        matching_task_id: Optional[str] = None,
    ) -> None:
        """Set/update MultiCa column matching results (update of existing state).

        Used to update an existing matching state. Pass matching_task_id from
        start_multica_matching to update that matching. The API requires matching
        to be a non-empty dict.

        Args:
            session_uuid (str): Session UUID
            matching (Dict[str, str]): Column mapping (target column -> source column).
                Must be non-empty. Use "" for columns with no match.
            matching_task_id (Optional[str]): Matching task ID from start_multica_matching
                (required to update existing).

        Raises:
            ValidationError: If matching is empty (400).
        """
        if not matching:
            raise ValidationError(
                message="'matching' must be a non-empty dictionary",
                status_code=400,
                response_data=None,
            )
        path = self._contract_path("cd.set_multica_matching", uuid=session_uuid)
        payload = CDMatchSetRequestContract(matching=matching, matching_task_id=matching_task_id)
        json_data = payload.model_dump(exclude_none=True)
        await self._make_request("POST", path, json_data=json_data)

    async def run_multica(
        self,
        uploaded_data: UploadedData,
        matching_task_id: str,
        threshold: Optional[float] = 0.01,
        roots: Optional[List[str]] = None,
        sinks: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit a MultiCa causal discovery task.

        Args:
            uploaded_data (UploadedData): From upload_data_for_multica.
            matching_task_id (str): Matching task ID from start_multica_matching (required).
            threshold (Optional[float]): Significance threshold (default: 0.01).
                Use ``None`` to omit threshold and let backend defaults apply.
            roots (Optional[List[str]]): Optional list of root variables
            sinks (Optional[List[str]]): Optional list of sink variables
            params (Optional[Dict]): Optional algorithm params (e.g. max_iter)

        Returns:
            Dict: Task response (status: queued)

        Raises:
            ValidationError: If the request payload does not align with the contract (400).
        """
        ref = uploaded_data
        try:
            payload = CDMulticaRunRequestContract(
                task_id=ref.task_id,
                s3_key=ref.s3_key,
                threshold=threshold,
                roots=roots,
                sinks=sinks,
                params=params,
                matching_task_id=matching_task_id,
            )
        except PydanticValidationError as e:
            raise ValidationError(
                message=str(e),
                status_code=400,
                response_data=None,
            ) from e
        path = self._contract_path("cd.run_multica", uuid=ref.session_uuid)
        json_data = payload.model_dump(exclude_none=True)
        response = await self._make_request("POST", path, json_data=json_data)
        return response

    async def delete_multica_matching(
        self, session_uuid: str, matching_task_id: Optional[str] = None
    ) -> None:
        """Delete MultiCa matching state and S3 artifacts for the session.

        Corresponds to DELETE /cd/match/{session_uuid}
        ?matching_type=multica&matching_task_id=...
        Use after run_multica to clean up matching state.

        Args:
            session_uuid (str): Session UUID
            matching_task_id (Optional[str]): Matching task ID from start_multica_matching
                (required by API)

        Raises:
            NotFoundError: If matching state does not exist (404).  # noqa: DAR402
            APIError: If matching is still pending (409).  # noqa: DAR402
        """
        await self.delete_matching(session_uuid, "multica", matching_task_id)
