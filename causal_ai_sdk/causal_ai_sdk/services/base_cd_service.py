"""Base Causal Discovery service with common functionality."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx
from causal_ai_sdk.contracts import get_contract
from causal_ai_sdk.exceptions import APIError, ValidationError
from causal_ai_sdk.models.cd import CDUploadURL, UploadedData
from causal_ai_sdk.services.base import BaseService
from causal_ai_sdk.utils.polling import poll_until_ready_or_fail
from pydantic import ValidationError as PydanticValidationError


class BaseCDService(BaseService):
    """Base class for Causal Discovery services using Template Method pattern.

    This class provides common functionality shared by MultiCa and TraCKR services,
    while allowing subclasses to customize specific behavior through template methods.
    """

    def __init__(self, config, http_client):
        """Initialize base CD service.

        Args:
            config (Config): Configuration instance
            http_client (HTTPClient): HTTP client instance
        """
        super().__init__(config, http_client)

    @staticmethod
    def _contract_path(name: str, **path_params: str) -> str:
        """Build endpoint path from the contract registry.

        Args:
            name (str): Contract registry key.
            **path_params (str): Named path parameters for endpoint formatting.

        Returns:
            str: Formatted endpoint path.
        """
        return get_contract(name).path.format(**path_params)

    async def get_upload_url(self, session_uuid: str) -> CDUploadURL:
        """Get presigned URL for uploading causal discovery data to S3.

        Args:
            session_uuid (str): Session UUID

        Returns:
            CDUploadURL: Upload URL instance with presigned URL, S3 key, and task_id.

        Raises:
            ValidationError: If response is missing required keys or has invalid types.
        """
        path = self._contract_path("cd.get_upload_url", uuid=session_uuid)
        response = await self._make_request("POST", path)
        try:
            return CDUploadURL.model_validate(response)
        except PydanticValidationError as e:
            raise ValidationError(
                message=str(e),
                status_code=None,
                response_data=None,
            ) from e

    async def wait_for_task(
        self,
        uploaded_data: Union[UploadedData, str],
        session_uuid: Optional[str] = None,
        timeout: int = 300,
        interval: int = 5,
    ) -> Dict[str, Any]:
        """Wait for task to complete (with automatic polling).

        Args:
            uploaded_data (Union[UploadedData, str]): UploadedData instance or task_id string.
            session_uuid (Optional[str]): Session UUID (required if uploaded_data is task_id).
            timeout (int): Maximum time to wait in seconds (default: 300)
            interval (int): Time between polls in seconds (default: 5)

        Returns:
            Dict: Final task status

        Raises:
            ValidationError: If parameters are invalid or task fails
        """
        if isinstance(uploaded_data, UploadedData):
            task_id = uploaded_data.task_id
            session_uuid = uploaded_data.session_uuid
        elif isinstance(uploaded_data, str):
            task_id = uploaded_data
            if session_uuid is None:
                raise ValidationError(
                    message="session_uuid is required when uploaded_data is a task_id string",
                    status_code=None,
                    response_data=None,
                )
        else:
            raise ValidationError(
                message="uploaded_data must be UploadedData instance or task_id string",
                status_code=None,
                response_data=None,
            )

        async def check() -> Dict[str, Any]:
            return await self.get_task_status(session_uuid, task_id)

        def on_failed(status: Dict[str, Any]) -> None:
            error_msg = status.get("error") or "Unknown error"
            raise ValidationError(
                message=f"Task failed: {error_msg}",
                status_code=None,
                response_data=None,
            )

        return await poll_until_ready_or_fail(
            check_func=check,
            is_ready=lambda s: s.get("status") == "succeeded",
            is_failed=lambda s: s.get("status") == "failed",
            on_failed=on_failed,
            timeout=timeout,
            interval=interval,
            timeout_error_message=f"Task did not complete within timeout ({timeout}s)",
            retry_exceptions=(APIError,),
            retry_if=lambda e: getattr(e, "status_code", None) == 500,
        )

    async def get_task_status(self, session_uuid: str, task_id: str) -> Dict[str, Any]:
        """Get the status of a causal discovery task.

        Args:
            session_uuid (str): Session UUID
            task_id (str): Task ID

        Returns:
            Dict: Task status with status, optional error
        """
        path = self._contract_path("cd.get_task_status", uuid=session_uuid)
        params = {"task_id": task_id}
        response = await self._make_request("GET", path, params=params)
        return response

    async def get_task_result(self, session_uuid: str, task_id: str) -> Dict[str, Any]:
        """Get the result URL for a completed causal discovery task.

        Args:
            session_uuid (str): Session UUID
            task_id (str): Task ID

        Returns:
            Dict: Task result with result_url
        """
        path = self._contract_path("cd.get_task_result", uuid=session_uuid)
        params = {"task_id": task_id}
        response = await self._make_request("GET", path, params=params)
        return response

    async def delete_task(self, session_uuid: str, task_id: str) -> None:
        """Delete a causal discovery task and associated resources.

        Args:
            session_uuid (str): Session UUID
            task_id (str): Task ID

        Raises:
            NotFoundError: If the task does not exist (404).  # noqa: DAR402
        """
        path = self._contract_path("cd.delete_task", uuid=session_uuid)
        params = {"task_id": task_id}
        await self._make_request("DELETE", path, params=params)

    async def delete_matching(
        self,
        session_uuid: str,
        matching_type: Literal["trackr", "multica"],
        matching_task_id: Optional[str] = None,
    ) -> None:
        """Delete matching state and S3 artifacts for a session.

        Corresponds to DELETE /cd/match/{session_uuid}
        ?matching_type=trackr|multica&matching_task_id=...
        Use after run to clean up matching state (e.g. trackr or multica).

        Args:
            session_uuid (str): Session UUID
            matching_type (Literal["trackr", "multica"]): Which matching state to delete
            matching_task_id (Optional[str]): Matching task ID from start_*_matching
                (required by API)

        Raises:
            NotFoundError: If matching state does not exist (404).  # noqa: DAR402
            APIError: If matching is still pending (409).  # noqa: DAR402
        """
        path = self._contract_path("cd.delete_matching", uuid=session_uuid)
        params: Dict[str, str] = {"matching_type": matching_type}
        if matching_task_id:
            params["matching_task_id"] = matching_task_id
        await self._make_request("DELETE", path, params=params)

    def _validate_file_format(self, file_path: Path) -> None:
        """Validate that a file has a supported format.

        Args:
            file_path (Path): Path to the file

        Raises:
            ValidationError: If file format is not supported
        """
        supported_extensions = {".json", ".csv"}
        if file_path.suffix.lower() not in supported_extensions:
            raise ValidationError(
                message=(
                    f"Unsupported file format: {file_path.suffix}. " "Supported formats: JSON, CSV"
                ),
                status_code=None,
                response_data=None,
            )

    async def _upload_bytes_to_presigned_url(
        self,
        upload_url: str,
        body: bytes,
        content_type: str,
    ) -> None:
        """Upload raw bytes to a presigned S3 URL.

        Args:
            upload_url (str): Presigned PUT URL
            body (bytes): Raw body to upload
            content_type (str): Content-Type header value (e.g. application/json, text/csv)

        Raises:
            ValidationError: If upload fails
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    upload_url,
                    content=body,
                    headers={"Content-Type": content_type},
                    timeout=30.0,
                )
                response.raise_for_status()
        except httpx.HTTPError as e:
            raise ValidationError(
                message=f"Failed to upload to S3: {str(e)}",
                status_code=None,
                response_data=None,
            ) from e

    @staticmethod
    def _content_type_for_path(file_path: Path) -> str:
        """Return Content-Type for a file path based on extension.

        Args:
            file_path (Path): Path to the file.

        Returns:
            Content-Type header value (e.g. application/json, text/csv).
        """
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            return "application/json"
        if suffix == ".csv":
            return "text/csv"
        return "application/octet-stream"

    async def _upload_data_internal(
        self,
        session_uuid: str,
        file_paths: List[Path],
    ) -> tuple[CDUploadURL, None]:
        """Upload file(s) to S3 as raw bytes or ZIP (template method).

        1. Validate upload requirements (delegated to subclass)
        2. Build upload body (bytes) and Content-Type (delegated to subclass)
        3. Get upload URL
        4. PUT body to presigned URL

        Args:
            session_uuid (str): Session UUID
            file_paths (List[Path]): List of file paths

        Returns:
            tuple[CDUploadURL, None]: Upload URL object; second element unused (for API compat).
        """
        self._validate_upload_requirements(file_paths)
        body, content_type = self._build_upload_body(file_paths)
        upload_url = await self.get_upload_url(session_uuid)
        await self._upload_bytes_to_presigned_url(upload_url.upload_url, body, content_type)
        return upload_url, None

    def _validate_upload_requirements(self, file_paths: List[Path]) -> Dict[Path, Any]:
        """Validate upload requirements (template method).

        Args:
            file_paths (List[Path]): List of file paths to validate

        Returns:
            Dict[Path, Any]: Optional cache (e.g. for MultiCa single JSON).  # noqa: DAR202

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _validate_upload_requirements")

    def _build_upload_body(self, file_paths: List[Path]) -> Tuple[bytes, str]:
        """Build upload body bytes and Content-Type (template method).

        Args:
            file_paths (List[Path]): List of file paths (already validated).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _build_upload_body")
