"""Unit tests for SDK MultiCa (CD) service."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock

import pytest
from causal_ai_sdk.config import Config
from causal_ai_sdk.exceptions import ValidationError
from causal_ai_sdk.models.cd import UploadedData
from causal_ai_sdk.services.multica_service import MultiCaService


def _make_multica_service(mock_http=None):
    if mock_http is None:
        mock_http = AsyncMock()
    config = Config(api_key="test-key", base_url="https://api.test")
    return MultiCaService(config, mock_http), mock_http


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def test_start_multica_matching_calls_post_with_correct_path_and_body():
    """start_multica_matching POSTs to /cd/multica/match/{uuid} with s3_key and optional params."""
    service, mock_http = _make_multica_service()
    mock_http.request = AsyncMock(return_value={"matching_task_id": "m1", "status": "pending"})

    uploaded = UploadedData(
        task_id="task-1",
        session_uuid="s1",
        s3_key="cd/s1/task-1/input",
    )

    result = _run(
        service.start_multica_matching(
            uploaded,
            metadata=[{"dataset": "d1"}],
            params={"gamma": 0.5},
        )
    )

    assert result["status"] == "pending"
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "POST"
    assert "/cd/multica/match/s1" in call_kw["url"]
    assert call_kw["json_data"]["s3_key"] == "cd/s1/task-1/input"
    assert call_kw["json_data"]["metadata"] == [{"dataset": "d1"}]
    assert call_kw["json_data"]["params"] == {"gamma": 0.5}


def test_run_multica_calls_post_with_correct_path_and_body():
    """run_multica POSTs to /cd/multica/run/{uuid} with task_id, s3_key, matching_task_id, etc."""
    service, mock_http = _make_multica_service()
    mock_http.request = AsyncMock(return_value={"uuid": "s1", "task_id": "t1", "status": "queued"})

    uploaded = UploadedData(
        task_id="task-1",
        session_uuid="s1",
        s3_key="cd/s1/task-1/input",
    )

    result = _run(
        service.run_multica(
            uploaded,
            matching_task_id="match-1",
            threshold=0.02,
            roots=["X"],
            sinks=["Y"],
        )
    )

    assert result["status"] == "queued"
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "POST"
    assert "/cd/multica/run/s1" in call_kw["url"]
    assert call_kw["json_data"]["task_id"] == "task-1"
    assert call_kw["json_data"]["s3_key"] == "cd/s1/task-1/input"
    assert call_kw["json_data"]["matching_task_id"] == "match-1"
    assert call_kw["json_data"]["threshold"] == 0.02
    assert call_kw["json_data"]["roots"] == ["X"]
    assert call_kw["json_data"]["sinks"] == ["Y"]


def test_run_multica_requires_matching_task_id():
    """run_multica requires matching_task_id (required by API); omitting it raises TypeError."""
    service, _ = _make_multica_service()
    uploaded = UploadedData(
        task_id="task-1",
        session_uuid="s1",
        s3_key="cd/s1/task-1/input",
    )
    with pytest.raises(TypeError, match="matching_task_id"):
        _run(service.run_multica(uploaded, threshold=0.01))


def test_get_multica_matching_calls_get_with_correct_path():
    """get_multica_matching sends GET to /cd/multica/match/{uuid}."""
    service, mock_http = _make_multica_service()
    mock_http.request = AsyncMock(
        return_value={"status": "completed", "current_matched": {"A": "B"}}
    )

    result = _run(service.get_multica_matching("s1"))

    assert result["status"] == "completed"
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "GET"
    assert "/cd/multica/match/s1" in call_kw["url"]


def test_get_multica_matching_passes_matching_task_id_in_params():
    """get_multica_matching sends matching_task_id as query param when provided."""
    service, mock_http = _make_multica_service()
    mock_http.request = AsyncMock(
        return_value={"status": "completed", "current_matched": {"A": "B"}}
    )

    _run(service.get_multica_matching("s1", matching_task_id="match-456"))

    call_kw = mock_http.request.await_args.kwargs
    assert "/cd/multica/match/s1" in call_kw["url"]
    assert call_kw.get("params") == {"matching_task_id": "match-456"}


def test_set_multica_matching_calls_post_with_matching_body():
    """set_multica_matching POSTs to /cd/multica/match/{uuid}/set with matching dict."""
    service, mock_http = _make_multica_service()
    mock_http.request = AsyncMock(return_value={})

    _run(service.set_multica_matching("s1", matching={"col_a": "src_a", "col_b": "src_b"}))

    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "POST"
    assert "/cd/multica/match/s1/set" in call_kw["url"]
    assert call_kw["json_data"]["matching"] == {"col_a": "src_a", "col_b": "src_b"}


def test_set_multica_matching_includes_matching_task_id_in_body():
    """set_multica_matching request body includes matching_task_id when provided."""
    service, mock_http = _make_multica_service()
    mock_http.request = AsyncMock(return_value={})

    _run(
        service.set_multica_matching(
            "s1",
            matching={"col_a": "src_a"},
            matching_task_id="match-789",
        )
    )

    call_kw = mock_http.request.await_args.kwargs
    assert "/cd/multica/match/s1/set" in call_kw["url"]
    assert call_kw["json_data"]["matching_task_id"] == "match-789"


def test_set_multica_matching_raises_when_matching_empty():
    """set_multica_matching raises ValidationError when matching is empty."""
    service, _ = _make_multica_service()

    with pytest.raises(ValidationError, match="matching.*non-empty|non-empty.*matching"):
        _run(service.set_multica_matching("s1", matching={}))


def test_delete_multica_matching_passes_matching_task_id_in_params():
    """delete_multica_matching sends matching_task_id as query param when provided."""
    service, mock_http = _make_multica_service()
    mock_http.request = AsyncMock(return_value={})

    _run(service.delete_multica_matching("s1", matching_task_id="match-del"))

    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "DELETE"
    assert "/cd/match/s1" in call_kw["url"]
    assert call_kw.get("params") == {
        "matching_type": "multica",
        "matching_task_id": "match-del",
    }


def test_wait_for_matching_passes_matching_task_id_to_get():
    """wait_for_matching passes matching_task_id to get_multica_matching (query params)."""
    service, mock_http = _make_multica_service()
    mock_http.request = AsyncMock(return_value={"status": "completed", "current_matched": {}})

    _run(
        service.wait_for_matching(
            "s1",
            timeout=10,
            interval=1,
            matching_task_id="match-wait",
        )
    )

    call_kw = mock_http.request.await_args.kwargs
    assert "/cd/multica/match/s1" in call_kw["url"]
    assert call_kw.get("params") == {"matching_task_id": "match-wait"}
