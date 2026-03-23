"""Causal Discovery service (Facade pattern)."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from causal_ai_sdk.exceptions import ValidationError
from causal_ai_sdk.models.cd import CDUploadURL, UploadedData
from causal_ai_sdk.services.base import BaseService
from causal_ai_sdk.services.lingam_service import LingamService
from causal_ai_sdk.services.multica_service import MultiCaService
from causal_ai_sdk.services.trackr_service import TraCKRService


class CDService(BaseService):
    """Causal Discovery service facade for running MultiCa, TraCKR, and LiNGAM.

    This service acts as a facade, delegating to MultiCaService, TraCKRService,
    and LingamService while maintaining backward compatibility with the existing API.

    This service provides methods for uploading data, managing column matching,
    and executing causal discovery tasks.
    """

    def __init__(self, config, http_client):
        """Initialize CD service.

        Args:
            config (Config): Configuration instance
            http_client (HTTPClient): HTTP client instance
        """
        super().__init__(config, http_client)
        # Initialize sub-services
        self._multica_service = MultiCaService(config, http_client)
        self._trackr_service = TraCKRService(config, http_client)
        self._lingam_service = LingamService(config, http_client)

    async def get_upload_url(self, session_uuid: str) -> CDUploadURL:
        """Get presigned URL for uploading causal discovery data to S3.

        Args:
            session_uuid (str): Session UUID

        Returns:
            CDUploadURL: Upload URL instance with presigned URL, S3 key, and task_id.
        """
        return await self._multica_service.get_upload_url(session_uuid)

    async def upload_data_for_multica(
        self,
        session_uuid: str,
        file_paths: Union[str, Path, List[Union[str, Path]]],
    ) -> UploadedData:
        """Upload multiple datasets for MultiCa causal discovery.

        This method automatically:
        1. Validates that at least 2 files are provided (or one JSON with multiple datasets)
        2. Validates that all files are in supported formats (JSON, CSV)
        3. Gets a presigned upload URL
        4. Uploads raw file bytes (single JSON/CSV) or a ZIP of files to S3

        Args:
            session_uuid (str): Session UUID
            file_paths (Union[str, Path, List[Union[str, Path]]]):
                Single file path or list of file paths (CSV or JSON)

        Returns:
            UploadedData: Uploaded data reference (task_id, session_uuid, s3_key).
        """
        return await self._multica_service.upload_data_for_multica(session_uuid, file_paths)

    async def upload_data_for_trackr(
        self,
        session_uuid: str,
        file_path: Union[str, Path],
    ) -> UploadedData:
        """Upload single dataset for TraCKR causal discovery.

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
        return await self._trackr_service.upload_data_for_trackr(session_uuid, file_path)

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
        return await self._lingam_service.upload_data_for_lingam(session_uuid, file_path)

    async def wait_for_matching(
        self,
        session_uuid: str,
        mode: str = "multica",
        timeout: int = 300,
        interval: int = 3,
        matching_task_id: Optional[str] = None,
        retry_on_5xx: bool = True,
    ) -> Dict[str, Any]:
        """Wait for matching computation to complete (with automatic polling).

        Args:
            session_uuid (str): Session UUID
            mode (str): Matching mode ("multica" or "trackr", default: "multica")
            timeout (int): Maximum time to wait in seconds (default: 300)
            interval (int): Time between polls in seconds (default: 3)
            matching_task_id (Optional[str]): Matching task ID from start_*_matching
                (required for API)
            retry_on_5xx (bool): If True, retry on 500/503; if False, raise on first 5xx
                (default: True).

        Returns:
            Dict: Final matching result

        Raises:
            ValidationError: If mode is invalid
        """
        if mode == "multica":
            return await self._multica_service.wait_for_matching(
                session_uuid, timeout, interval, matching_task_id, retry_on_5xx=retry_on_5xx
            )
        elif mode == "trackr":
            return await self._trackr_service.wait_for_matching(
                session_uuid, timeout, interval, matching_task_id
            )
        else:
            raise ValidationError(
                message=f"Invalid mode: {mode}. Must be 'multica' or 'trackr'",
                status_code=None,
                response_data=None,
            )

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
            session_uuid (Optional[str]): Session UUID (required if uploaded_data is task_id string)
            timeout (int): Maximum time to wait in seconds (default: 300)
            interval (int): Time between polls in seconds (default: 5)

        Returns:
            Dict: Final task status
        """
        # Delegate to either service (both have the same implementation)
        return await self._multica_service.wait_for_task(
            uploaded_data, session_uuid, timeout, interval
        )

    async def get_task_status(self, session_uuid: str, task_id: str) -> Dict[str, Any]:
        """Get the status of a causal discovery task.

        Args:
            session_uuid (str): Session UUID
            task_id (str): Task ID

        Returns:
            Dict: Task status with status, optional error
        """
        # Delegate to either service (both have the same implementation)
        return await self._multica_service.get_task_status(session_uuid, task_id)

    async def get_task_result(self, session_uuid: str, task_id: str) -> Dict[str, Any]:
        """Get the result URL for a completed causal discovery task.

        Args:
            session_uuid (str): Session UUID
            task_id (str): Task ID

        Returns:
            Dict: Task result with result_url
        """
        # Delegate to either service (both have the same implementation)
        return await self._multica_service.get_task_result(session_uuid, task_id)

    async def delete_task(self, session_uuid: str, task_id: str) -> None:
        """Delete a causal discovery task and associated resources.

        Args:
            session_uuid (str): Session UUID
            task_id (str): Task ID

        Returns:
            None

        Raises:
            NotFoundError: If the task does not exist (404).  # noqa: DAR402
        """
        # Delegate to either service (both have the same implementation)
        return await self._multica_service.delete_task(session_uuid, task_id)

    async def delete_matching(
        self,
        session_uuid: str,
        matching_type: Literal["trackr", "multica"],
        matching_task_id: Optional[str] = None,
    ) -> None:
        """Delete matching state and S3 artifacts for a session.

        Corresponds to DELETE /cd/match/{session_uuid}
        ?matching_type=trackr|multica&matching_task_id=...
        Use after run to clean up matching state.

        Args:
            session_uuid (str): Session UUID
            matching_type (Literal["trackr", "multica"]): Which matching state to delete
            matching_task_id (Optional[str]): Matching task ID from start_*_matching
                (required by API)

        Raises:
            NotFoundError: If matching state does not exist (404).  # noqa: DAR402
            APIError: If matching is still pending (409).  # noqa: DAR402
        """
        await self._multica_service.delete_matching(session_uuid, matching_type, matching_task_id)

    async def delete_trackr_matching(
        self, session_uuid: str, matching_task_id: Optional[str] = None
    ) -> None:
        """Delete TraCKR matching state for the session (convenience for matching_type=trackr).

        Args:
            session_uuid (str): Session UUID.
            matching_task_id (Optional[str]): Matching task ID from start_trackr_matching
                (required by API).
        """
        await self.delete_matching(session_uuid, "trackr", matching_task_id)

    async def delete_multica_matching(
        self, session_uuid: str, matching_task_id: Optional[str] = None
    ) -> None:
        """Delete MultiCa matching state for the session (convenience for matching_type=multica).

        Args:
            session_uuid (str): Session UUID.
            matching_task_id (Optional[str]): Matching task ID from start_multica_matching
                (required by API).
        """
        await self.delete_matching(session_uuid, "multica", matching_task_id)

    # MultiCa methods - delegate to MultiCaService

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
        return await self._multica_service.start_multica_matching(uploaded_data, metadata, params)

    async def get_multica_matching(self, session_uuid: str) -> Dict[str, Any]:
        """Get MultiCa column matching results.

        Args:
            session_uuid (str): Session UUID

        Returns:
            Dict: Matching result with current_matched, match_rate, etc.
        """
        return await self._multica_service.get_multica_matching(session_uuid)

    async def set_multica_matching(
        self,
        session_uuid: str,
        matching: Dict[str, str],
        matching_task_id: Optional[str] = None,
    ) -> None:
        """Set/update MultiCa column matching results.

        Args:
            session_uuid (str): Session UUID
            matching (Dict[str, str]): Column mapping (target column -> source column)
            matching_task_id (Optional[str]): Matching task ID from start_multica_matching
                (required to update existing)
        """
        await self._multica_service.set_multica_matching(session_uuid, matching, matching_task_id)

    async def run_multica(
        self,
        uploaded_data: UploadedData,
        matching_task_id: str,
        threshold: Optional[float] = 0.01,
        roots: Optional[List[str]] = None,
        sinks: Optional[List[str]] = None,
        params: Optional[Dict] = None,
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
            ValidationError: If the request payload does not align with the
                contract (400).  # noqa: DAR402
        """
        return await self._multica_service.run_multica(
            uploaded_data, matching_task_id, threshold, roots, sinks, params
        )

    # TraCKR methods - delegate to TraCKRService

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
        return await self._trackr_service.start_trackr_matching(
            uploaded_data, source_kg_id, target_metadata, source_metadata, params
        )

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
        return await self._trackr_service.get_trackr_matching(session_uuid, matching_task_id)

    async def set_trackr_matching(
        self,
        session_uuid: str,
        matching: Dict[str, str],
        matching_task_id: Optional[str] = None,
    ) -> None:
        """Set/update TraCKR column matching results.

        Args:
            session_uuid (str): Session UUID
            matching (Dict[str, str]): Column mapping (target column -> source column)
            matching_task_id (Optional[str]): Matching task ID from start_trackr_matching
                (required to update existing)
        """
        await self._trackr_service.set_trackr_matching(session_uuid, matching, matching_task_id)

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
            ValidationError: If the request payload does not align with the
                contract (400).  # noqa: DAR402
            NotFoundError: If the knowledge graph does not exist (404).  # noqa: DAR402
            APIError: If KG has no S3 reference or validation fails (500).  # noqa: DAR402
        """
        return await self._trackr_service.run_trackr(
            uploaded_data,
            transferred_knowledge,
            matching_task_id,
            threshold,
            params,
        )

    async def run_lingam(
        self,
        uploaded_data: UploadedData,
        threshold: Optional[float] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Submit a LayeredLiNGAM causal discovery task.

        Args:
            uploaded_data (UploadedData): From upload_data_for_lingam.
            threshold (Optional[float]): Significance threshold (default: None).
                Use ``None`` to omit threshold and let backend defaults apply.
            params (Optional[Dict]): Optional Lingam params

        Returns:
            Dict: Task response (status: queued)
        """
        return await self._lingam_service.run_lingam(uploaded_data, threshold, params)
