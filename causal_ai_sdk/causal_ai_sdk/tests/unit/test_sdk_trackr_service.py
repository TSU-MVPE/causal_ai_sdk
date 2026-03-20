"""Unit tests for SDK TraCKR (CD) service."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock

import pytest
from causal_ai_sdk.config import Config
from causal_ai_sdk.exceptions import ValidationError
from causal_ai_sdk.models.cd import UploadedData
from causal_ai_sdk.services.trackr_service import TraCKRService


def _make_trackr_service(mock_http=None):
    if mock_http is None:
        mock_http = AsyncMock()
    config = Config(api_key="test-key", base_url="https://api.test")
    return TraCKRService(config, mock_http), mock_http


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def test_start_trackr_matching_calls_post_with_correct_path_and_body():
    """start_trackr_matching POSTs to /cd/trackr/match/{uuid} with target_s3_key, source_kg_id."""
    service, mock_http = _make_trackr_service()
    mock_http.request = AsyncMock(return_value={"matching_task_id": "m1", "status": "pending"})

    uploaded = UploadedData(
        task_id="task-1",
        session_uuid="s1",
        s3_key="cd/s1/task-1/input",
    )

    result = _run(
        service.start_trackr_matching(
            uploaded,
            source_kg_id="kg-123",
            target_metadata={"A": "desc"},
            source_metadata={"B": "desc"},
            params={"gamma": 0.5},
        )
    )

    assert result["status"] == "pending"
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "POST"
    assert "/cd/trackr/match/s1" in call_kw["url"]
    assert call_kw["json_data"]["target_s3_key"] == "cd/s1/task-1/input"
    assert call_kw["json_data"]["source_kg_id"] == "kg-123"
    assert call_kw["json_data"]["target_metadata"] == {"A": "desc"}
    assert call_kw["json_data"]["source_metadata"] == {"B": "desc"}
    assert call_kw["json_data"]["params"] == {"gamma": 0.5}


def test_run_trackr_calls_post_with_correct_path_and_body():
    """run_trackr POSTs to /cd/trackr/run/{uuid} with task_id, s3_key, matching_task_id, etc."""
    service, mock_http = _make_trackr_service()
    mock_http.request = AsyncMock(return_value={"uuid": "s1", "task_id": "t1", "status": "queued"})

    uploaded = UploadedData(
        task_id="task-1",
        session_uuid="s1",
        s3_key="cd/s1/task-1/input",
    )
    transferred = {"session_uuid": "kg-session", "kg_id": "kg-1"}

    result = _run(
        service.run_trackr(
            uploaded,
            matching_task_id="match-1",
            transferred_knowledge=transferred,
            threshold=0.02,
            params={"alpha": 0.1},
        )
    )

    assert result["status"] == "queued"
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "POST"
    assert "/cd/trackr/run/s1" in call_kw["url"]
    assert call_kw["json_data"]["task_id"] == "task-1"
    assert call_kw["json_data"]["s3_key"] == "cd/s1/task-1/input"
    assert call_kw["json_data"]["matching_task_id"] == "match-1"
    assert call_kw["json_data"]["transferred_knowledge"] == transferred
    assert call_kw["json_data"]["threshold"] == 0.02
    assert call_kw["json_data"]["params"] == {"alpha": 0.1}


def test_run_trackr_requires_matching_task_id():
    """run_trackr requires matching_task_id (required by API); omitting it raises TypeError."""
    service, _ = _make_trackr_service()
    uploaded = UploadedData(
        task_id="task-1",
        session_uuid="s1",
        s3_key="cd/s1/task-1/input",
    )
    transferred = {"session_uuid": "s1", "kg_id": "kg-1"}
    with pytest.raises(TypeError, match="matching_task_id"):
        _run(
            service.run_trackr(
                uploaded,
                transferred_knowledge=transferred,
                threshold=0.01,
            )
        )


def test_get_trackr_matching_calls_get_with_correct_path():
    """get_trackr_matching sends GET to /cd/trackr/match/{uuid}."""
    service, mock_http = _make_trackr_service()
    mock_http.request = AsyncMock(return_value={"status": "completed", "knowledge_coverage": 0.9})

    result = _run(service.get_trackr_matching("s1"))

    assert result["status"] == "completed"
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "GET"
    assert "/cd/trackr/match/s1" in call_kw["url"]


def test_get_trackr_matching_passes_matching_task_id_in_params():
    """get_trackr_matching sends matching_task_id as query param when provided."""
    service, mock_http = _make_trackr_service()
    mock_http.request = AsyncMock(return_value={"status": "completed", "knowledge_coverage": 0.9})

    _run(service.get_trackr_matching("s1", matching_task_id="match-456"))

    call_kw = mock_http.request.await_args.kwargs
    assert "/cd/trackr/match/s1" in call_kw["url"]
    assert call_kw.get("params") == {"matching_task_id": "match-456"}


def test_set_trackr_matching_calls_post_with_matching_body():
    """set_trackr_matching POSTs to /cd/trackr/match/{uuid}/set with matching dict."""
    service, mock_http = _make_trackr_service()
    mock_http.request = AsyncMock(return_value={})

    _run(service.set_trackr_matching("s1", matching={"col_a": "src_a", "col_b": "src_b"}))

    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "POST"
    assert "/cd/trackr/match/s1/set" in call_kw["url"]
    assert call_kw["json_data"]["matching"] == {"col_a": "src_a", "col_b": "src_b"}


def test_set_trackr_matching_includes_matching_task_id_in_body():
    """set_trackr_matching request body includes matching_task_id when provided."""
    service, mock_http = _make_trackr_service()
    mock_http.request = AsyncMock(return_value={})

    _run(
        service.set_trackr_matching(
            "s1",
            matching={"col_a": "src_a"},
            matching_task_id="match-789",
        )
    )

    call_kw = mock_http.request.await_args.kwargs
    assert "/cd/trackr/match/s1/set" in call_kw["url"]
    assert call_kw["json_data"]["matching_task_id"] == "match-789"


def test_set_trackr_matching_raises_when_matching_empty():
    """set_trackr_matching raises ValidationError when matching is empty."""
    service, _ = _make_trackr_service()

    with pytest.raises(ValidationError, match="matching.*non-empty|non-empty.*matching"):
        _run(service.set_trackr_matching("s1", matching={}))


def test_delete_trackr_matching_passes_matching_task_id_in_params():
    """delete_trackr_matching sends matching_task_id as query param when provided."""
    service, mock_http = _make_trackr_service()
    mock_http.request = AsyncMock(return_value={})

    _run(service.delete_trackr_matching("s1", matching_task_id="match-del"))

    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "DELETE"
    assert "/cd/match/s1" in call_kw["url"]
    assert call_kw.get("params") == {
        "matching_type": "trackr",
        "matching_task_id": "match-del",
    }


def test_wait_for_matching_passes_matching_task_id_to_get():
    """wait_for_matching passes matching_task_id to get_trackr_matching (query params)."""
    service, mock_http = _make_trackr_service()
    mock_http.request = AsyncMock(return_value={"status": "completed", "knowledge_coverage": 0.9})

    _run(
        service.wait_for_matching(
            "s1",
            timeout=10,
            interval=1,
            matching_task_id="match-wait",
        )
    )

    call_kw = mock_http.request.await_args.kwargs
    assert "/cd/trackr/match/s1" in call_kw["url"]
    assert call_kw.get("params") == {"matching_task_id": "match-wait"}


def test_run_trackr_raises_when_transferred_knowledge_missing_kg_id():
    """run_trackr raises ValidationError when transferred_knowledge lacks session_uuid or kg_id."""
    service, _ = _make_trackr_service()

    uploaded = UploadedData(
        task_id="task-1",
        session_uuid="s1",
        s3_key="cd/s1/task-1/input",
    )
    matching_task_id = "match-1"

    with pytest.raises(ValidationError):
        _run(
            service.run_trackr(
                uploaded,
                transferred_knowledge={},
                matching_task_id=matching_task_id,
            )
        )

    with pytest.raises(ValidationError):
        _run(
            service.run_trackr(
                uploaded,
                transferred_knowledge={"session_uuid": "s"},
                matching_task_id=matching_task_id,
            )
        )

    with pytest.raises(ValidationError):
        _run(
            service.run_trackr(
                uploaded,
                transferred_knowledge={"kg_id": "k1"},
                matching_task_id=matching_task_id,
            )
        )
