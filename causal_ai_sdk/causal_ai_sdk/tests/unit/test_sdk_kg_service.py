"""Unit tests for SDK Knowledge Graph (KG) service."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock

from causal_ai_sdk.config import Config
from causal_ai_sdk.services.kg_service import KGService


def _make_kg_service(mock_http=None):
    if mock_http is None:
        mock_http = AsyncMock()
    config = Config(api_key="test-key", base_url="https://api.test")
    return KGService(config, mock_http), mock_http


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def test_init_session_calls_post_with_correct_path():
    """init_session POSTs to /kg/init."""
    service, mock_http = _make_kg_service()
    mock_http.request = AsyncMock(return_value={"uuid": "s1", "status": "ok"})

    result = _run(service.init_session())

    assert result["uuid"] == "s1"
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "POST"
    assert "/kg/init" in call_kw["url"]


def test_get_upload_url_calls_post_with_correct_path_and_body():
    """get_upload_url POSTs to /kg/upload-url/{uuid} with optional filename."""
    service, mock_http = _make_kg_service()
    mock_http.request = AsyncMock(
        return_value={
            "upload_url": "https://s3.example/presigned",
            "s3_key": "kg/s1/source.json",
            "expires_in": 3600,
        }
    )

    result = _run(service.get_upload_url("s1", filename="my_kg.json"))

    assert result["s3_key"] == "kg/s1/source.json"
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "POST"
    assert "/kg/upload-url/s1" in call_kw["url"]
    assert call_kw["json_data"]["filename"] == "my_kg.json"


def test_add_kg_calls_post_with_correct_path_and_body():
    """add_kg POSTs to /kg/add/{uuid} with title, columns, s3_key, optional row_count/size_bytes."""
    service, mock_http = _make_kg_service()
    mock_http.request = AsyncMock(return_value={"kg_id": "kg-123"})

    result = _run(
        service.add_kg(
            session_uuid="s1",
            title="My KG",
            columns=["A", "B", "C"],
            s3_key="kg/s1/source.json",
            row_count=100,
            size_bytes=2048,
        )
    )

    assert result["id"] == "kg-123"
    assert result["title"] == "My KG"
    assert result["num_nodes"] == 3
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "POST"
    assert "/kg/add/s1" in call_kw["url"]
    assert call_kw["json_data"]["title"] == "My KG"
    assert call_kw["json_data"]["columns"] == ["A", "B", "C"]
    assert call_kw["json_data"]["s3_key"] == "kg/s1/source.json"
    assert call_kw["json_data"]["row_count"] == 100
    assert call_kw["json_data"]["size_bytes"] == 2048


def test_list_kg_calls_get_with_correct_path():
    """list_kg sends GET to /kg/list/{uuid} and returns response['list']."""
    service, mock_http = _make_kg_service()
    mock_http.request = AsyncMock(
        return_value={"list": [{"kg_id": "k1", "title": "KG1"}, {"kg_id": "k2", "title": "KG2"}]}
    )

    result = _run(service.list_kg("s1"))

    assert len(result) == 2
    assert result[0]["kg_id"] == "k1"
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "GET"
    assert "/kg/list/s1" in call_kw["url"]


def test_get_kg_calls_get_with_correct_path_and_params():
    """get_kg sends GET to /kg/graph/{uuid}?gid=... and returns response['kg']."""
    service, mock_http = _make_kg_service()
    mock_http.request = AsyncMock(
        return_value={
            "kg": {
                "kg_id": "kg-1",
                "title": "My KG",
                "columns": ["A", "B"],
                "s3_key": "kg/s1/source.json",
            }
        }
    )

    result = _run(service.get_kg("s1", "kg-1"))

    assert result["kg_id"] == "kg-1"
    assert result["columns"] == ["A", "B"]
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "GET"
    assert "/kg/graph/s1" in call_kw["url"]
    assert call_kw["params"] == {"gid": "kg-1"}


def test_delete_session_calls_delete_with_correct_path():
    """delete_session sends DELETE to /kg/graph/{uuid}."""
    service, mock_http = _make_kg_service()
    mock_http.request = AsyncMock(return_value={})

    _run(service.delete_session("s1"))

    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "DELETE"
    assert "/kg/graph/s1" in call_kw["url"]


def test_delete_kg_session_calls_delete_session():
    """delete_kg_session is alias for delete_session."""
    service, mock_http = _make_kg_service()
    mock_http.request = AsyncMock(return_value={})

    _run(service.delete_kg_session("s2"))

    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "DELETE"
    assert "/kg/graph/s2" in call_kw["url"]
