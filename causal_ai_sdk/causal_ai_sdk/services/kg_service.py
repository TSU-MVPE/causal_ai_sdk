"""Knowledge Graph service."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import httpx
from causal_ai_sdk.contracts import get_contract
from causal_ai_sdk.contracts.requests import (
    KGAddRequestContract,
    KGUploadURLRequestContract,
)
from causal_ai_sdk.exceptions import ValidationError
from causal_ai_sdk.services.base import BaseService
from causal_ai_sdk.utils.dataset_schema import validate_columns_data


def _get_kg_metadata(file_path: Path) -> Tuple[List[str], int]:
    """Read file and return (columns, row_count) for KG registration.

    Args:
        file_path (Path): Path to CSV or JSON file.

    Returns:
        Tuple[List[str], int]: (column names, number of data rows)

    Raises:
        ValidationError: If format is unsupported or parsing fails.
    """
    suffix = file_path.suffix.lower()
    if suffix not in (".csv", ".json"):
        raise ValidationError(
            message=f"Unsupported file format: {suffix}. Use .csv or .json",
            status_code=None,
            response_data=None,
        )
    try:
        if suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return validate_columns_data(data, context="KG JSON file")
        # CSV
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            raise ValidationError(
                message="CSV file is empty",
                status_code=None,
                response_data=None,
            )
        columns = rows[0]
        data_rows = rows[1:]
        return validate_columns_data(
            {"columns": columns, "data": data_rows},
            context="KG CSV file",
        )
    except (json.JSONDecodeError, csv.Error) as e:
        raise ValidationError(
            message=f"Failed to parse file for metadata: {e}",
            status_code=None,
            response_data=None,
        ) from e


class KGService(BaseService):
    """Knowledge Graph service for managing causal knowledge graphs.

    This service provides methods for creating sessions, uploading knowledge
    graphs, and managing graph metadata.
    """

    def __init__(self, config, http_client):
        """Initialize KG service.

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

    async def init_session(self) -> Dict:
        """Initialize a new knowledge graph session.

        Returns:
            Dict: Response with uuid and status
        """
        path = self._contract_path("kg.init_session")
        response = await self._make_request("POST", path)
        return response

    async def get_upload_url(self, session_uuid: str, filename: Optional[str] = None) -> Dict:
        """Get presigned URL for uploading knowledge graph data to S3.

        Args:
            session_uuid (str): Session UUID
            filename (Optional[str]): Optional original filename

        Returns:
            Dict: Response with upload_url, s3_key, expires_in, etc.
        """
        path = self._contract_path("kg.get_upload_url", uuid=session_uuid)
        payload = KGUploadURLRequestContract(filename=filename)
        json_data = payload.model_dump(exclude_none=True)
        if not json_data:
            json_data = None
        response = await self._make_request("POST", path, json_data=json_data)
        return response

    async def add_kg(
        self,
        session_uuid: str,
        title: str,
        columns: List[str],
        s3_key: str,
        row_count: Optional[int] = None,
        size_bytes: Optional[int] = None,
    ) -> Dict:
        """Register knowledge graph metadata.

        Args:
            session_uuid (str): Session UUID
            title (str): Knowledge graph title
            columns (List[str]): List of column names
            s3_key (str): S3 key (obtained from get_upload_url)
            row_count (Optional[int]): Optional number of rows
            size_bytes (Optional[int]): Optional size in bytes

        Returns:
            Dict: Registered knowledge graph info (id, title, num_nodes)
        """
        path = self._contract_path("kg.add_kg", uuid=session_uuid)
        payload = KGAddRequestContract(
            title=title,
            columns=columns,
            s3_key=s3_key,
            row_count=row_count,
            size_bytes=size_bytes,
        )
        json_data: Dict[str, Union[str, List[str], int]] = payload.model_dump(exclude_none=True)

        response = await self._make_request("POST", path, json_data=json_data)
        return {
            "id": response["kg_id"],
            "title": title,
            "num_nodes": len(columns),
        }

    async def list_kg(self, session_uuid: str) -> List[Dict]:
        """List all knowledge graphs in a session.

        Args:
            session_uuid (str): Session UUID

        Returns:
            List[Dict]: List of knowledge graph items
        """
        path = self._contract_path("kg.list_kg", uuid=session_uuid)
        response = await self._make_request("GET", path)
        return list(response["list"])

    async def get_kg(self, session_uuid: str, kg_id: str) -> Dict:
        """Get detailed information about a specific knowledge graph.

        Args:
            session_uuid (str): Session UUID
            kg_id (str): Knowledge Graph ID

        Returns:
            Dict: Detailed knowledge graph information (columns, s3_key, download_url, etc.)
        """
        path = self._contract_path("kg.get_kg", uuid=session_uuid)
        params = {"gid": kg_id}
        response = await self._make_request("GET", path, params=params)
        return response["kg"]

    async def delete_session(self, session_uuid: str) -> None:
        """Delete a session and all associated knowledge graphs.

        Args:
            session_uuid (str): Session UUID
        """
        path = self._contract_path("kg.delete_session", uuid=session_uuid)
        await self._make_request("DELETE", path)

    async def delete_kg_session(self, session_uuid: str) -> None:
        """Alias for delete_session for clearer KG API semantics.

        Args:
            session_uuid (str): Session UUID
        """
        await self.delete_session(session_uuid)

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

    async def upload_kg_from_file(
        self,
        session_uuid: str,
        file_path: Union[str, Path],
        title: Optional[str] = None,
    ) -> Dict:
        """Upload knowledge graph from a CSV or JSON file.

        This method automatically:
        1. Reads file metadata (columns, row count) for registration
        2. Gets a presigned upload URL
        3. Uploads the file as raw bytes to S3
        4. Registers the knowledge graph metadata

        Title is taken from the optional argument or from the filename (stem).
        For CSV there is no title in the file; for JSON, a "title" key may exist
        but the SDK uses the filename stem or the title argument for registration.

        Args:
            session_uuid (str): Session UUID
            file_path (Union[str, Path]): Path to CSV or JSON (KG: header/columns + data rows)
            title (Optional[str]): Optional title (defaults to filename without extension)

        Returns:
            Dict: Registered knowledge graph info (id, title, num_nodes)

        Raises:
            ValidationError: If file is not .csv/.json or parsing fails
        """
        file_path = Path(file_path)
        if file_path.suffix.lower() not in (".json", ".csv"):
            raise ValidationError(
                message="Knowledge graph upload supports CSV or JSON only.",
                status_code=None,
                response_data=None,
            )

        # 1. Get metadata for registration (columns, row_count)
        try:
            columns, row_count = _get_kg_metadata(file_path)
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                message=f"Failed to read file metadata: {str(e)}",
                status_code=None,
                response_data=None,
            ) from e

        # 2. Get upload URL (async)
        upload_url_resp = await self.get_upload_url(session_uuid, filename=file_path.name)

        # 3. Upload file as raw bytes to S3
        body = file_path.read_bytes()
        content_type = self._content_type_for_path(file_path)
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.put(
                    upload_url_resp["upload_url"],
                    content=body,
                    headers={"Content-Type": content_type},
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
        except httpx.HTTPError as e:
            raise ValidationError(
                message=f"Failed to upload file to S3: {str(e)}",
                status_code=None,
                response_data=None,
            ) from e

        # 4. Register metadata (async)
        if title is None:
            title = file_path.stem
        size_bytes = len(body)

        return await self.add_kg(
            session_uuid=session_uuid,
            title=title,
            columns=columns,
            s3_key=upload_url_resp["s3_key"],
            row_count=row_count,
            size_bytes=size_bytes,
        )
