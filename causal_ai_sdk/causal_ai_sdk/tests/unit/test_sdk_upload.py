"""Unit tests: upload path uses raw file bytes (no converter)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from causal_ai_sdk.config import Config
from causal_ai_sdk.exceptions import ValidationError
from causal_ai_sdk.services.kg_service import KGService, _get_kg_metadata
from causal_ai_sdk.services.multica_service import MultiCaService
from causal_ai_sdk.services.trackr_service import TraCKRService


def _run(coro):
    return asyncio.run(coro)


def test_kg_upload_from_file_rejects_unsupported_format():
    """upload_kg_from_file raises ValidationError for non-CSV/non-JSON (e.g. .txt)."""
    config = Config(api_key="k", base_url="https://api.test")
    service = KGService(config, AsyncMock())
    with pytest.raises(ValidationError) as exc_info:
        _run(service.upload_kg_from_file("session-1", "/some/path/data.txt"))
    assert "CSV or JSON" in str(exc_info.value.message)


def test_kg_upload_from_file_sends_raw_csv_bytes(tmp_path: Path):
    """upload_kg_from_file sends CSV file bytes and text/csv (no conversion).

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    csv_file = tmp_path / "graph.csv"
    csv_file.write_text("A,B\n1,2\n3,4", encoding="utf-8")
    expected_bytes = csv_file.read_bytes()

    config = Config(api_key="k", base_url="https://api.test")
    mock_http = AsyncMock()
    mock_http.request = AsyncMock(
        side_effect=[
            {
                "uuid": "s1",
                "status": "ok",
                "s3_key": "kg/s1/key.csv",
                "upload_url": "https://s3.example/put",
                "expires_in": 3600,
            },
            {"kg_id": "kg-1"},
        ]
    )
    service = KGService(config, mock_http)
    mock_put = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = lambda: None
    mock_put.return_value = mock_resp
    with patch("causal_ai_sdk.services.kg_service.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.put = mock_put
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        result = _run(service.upload_kg_from_file("s1", csv_file))
    assert isinstance(result, dict)
    assert result.get("id") == "kg-1"
    mock_put.assert_awaited_once()
    assert mock_put.await_args is not None
    call_kw = mock_put.await_args.kwargs
    assert call_kw["content"] == expected_bytes
    assert call_kw["headers"]["Content-Type"] == "text/csv"


def test_kg_upload_from_file_sends_raw_json_bytes(tmp_path: Path):
    """upload_kg_from_file sends file bytes and application/json (no client-side conversion).

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    json_file = tmp_path / "graph.json"
    payload = {"columns": ["A", "B"], "data": [[1, 2], [3, 4]]}
    json_file.write_text(json.dumps(payload), encoding="utf-8")
    expected_bytes = json_file.read_bytes()

    config = Config(api_key="k", base_url="https://api.test")
    mock_http = AsyncMock()
    mock_http.request = AsyncMock(
        side_effect=[
            {
                "uuid": "s1",
                "status": "ok",
                "s3_key": "kg/s1/key.json",
                "upload_url": "https://s3.example/put",
                "expires_in": 3600,
            },
            {"kg_id": "kg-1"},
        ]
    )
    service = KGService(config, mock_http)

    mock_put = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = lambda: None
    mock_put.return_value = mock_resp

    with patch("causal_ai_sdk.services.kg_service.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.put = mock_put
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        result = _run(service.upload_kg_from_file("s1", json_file))

    assert isinstance(result, dict)
    assert result.get("id") == "kg-1"
    mock_put.assert_awaited_once()
    assert mock_put.await_args is not None
    call_kw = mock_put.await_args.kwargs
    assert call_kw["content"] == expected_bytes
    assert call_kw["headers"]["Content-Type"] == "application/json"


def test_trackr_build_upload_body_sends_raw_bytes(tmp_path: Path):
    """CD TraCKR _build_upload_body returns file bytes and correct Content-Type (no conversion).

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    config = Config(api_key="k", base_url="https://api.test")
    service = TraCKRService(config, AsyncMock())

    csv_file = tmp_path / "data.csv"
    csv_file.write_text("x,y\n1,2\n3,4", encoding="utf-8")
    body, content_type = service._build_upload_body([csv_file])
    assert body == csv_file.read_bytes()
    assert content_type == "text/csv"

    json_file = tmp_path / "data.json"
    json_file.write_text('{"columns":["x","y"],"data":[[1,2],[3,4]]}', encoding="utf-8")
    body2, content_type2 = service._build_upload_body([json_file])
    assert body2 == json_file.read_bytes()
    assert content_type2 == "application/json"


def test_kg_upload_from_file_rejects_invalid_json_schema_before_http(tmp_path: Path):
    """upload_kg_from_file raises ValidationError for invalid JSON schema without calling API.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    json_file = tmp_path / "bad.json"
    json_file.write_text('{"columns": "not-a-list", "data": []}', encoding="utf-8")
    config = Config(api_key="k", base_url="https://api.test")
    mock_http = AsyncMock()
    mock_http.request = AsyncMock()
    service = KGService(config, mock_http)
    with pytest.raises(ValidationError) as exc_info:
        _run(service.upload_kg_from_file("s1", json_file))
    assert "KG JSON file" in exc_info.value.message
    mock_http.request.assert_not_called()


def test_kg_get_metadata_rejects_json_with_columns_not_list(tmp_path: Path):
    """_get_kg_metadata raises ValidationError when JSON 'columns' is not a list.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    json_file = tmp_path / "bad.json"
    json_file.write_text('{"columns": "A,B", "data": [[1, 2]]}', encoding="utf-8")
    with pytest.raises(ValidationError) as exc_info:
        _get_kg_metadata(json_file)
    assert "KG JSON file" in exc_info.value.message
    assert "columns" in exc_info.value.message


def test_kg_get_metadata_rejects_json_with_row_length_mismatch(tmp_path: Path):
    """_get_kg_metadata raises ValidationError when a row has wrong length.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    json_file = tmp_path / "bad.json"
    json_file.write_text(
        '{"columns": ["A", "B"], "data": [[1, 2], [3, 4, 5]]}',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as exc_info:
        _get_kg_metadata(json_file)
    assert "KG JSON file" in exc_info.value.message
    assert "row" in exc_info.value.message


def test_kg_get_metadata_rejects_csv_with_row_length_mismatch(tmp_path: Path):
    """_get_kg_metadata raises ValidationError when CSV row has wrong column count.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("A,B\n1,2\n3,4,5", encoding="utf-8")
    with pytest.raises(ValidationError) as exc_info:
        _get_kg_metadata(csv_file)
    assert "KG CSV file" in exc_info.value.message
    assert "row" in exc_info.value.message


def test_multica_single_json_rejects_invalid_dataset(tmp_path: Path):
    """_validate_multica_files raises ValidationError when dataset has wrong row length.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    json_file = tmp_path / "multica.json"
    payload = {
        "data": [
            {"columns": ["A", "B"], "data": [[1, 2]]},
            {"columns": ["X", "Y"], "data": [[1, 2], [3, 4, 5]]},  # row length 3
        ],
    }
    json_file.write_text(json.dumps(payload), encoding="utf-8")
    config = Config(api_key="k", base_url="https://api.test")
    service = MultiCaService(config, AsyncMock())
    with pytest.raises(ValidationError) as exc_info:
        service._validate_multica_files([json_file])
    assert "JSON dataset at index 1" in exc_info.value.message
    assert "row" in exc_info.value.message
