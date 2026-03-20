"""Unit tests for SDK LiNGAM (CD) service."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock

import pytest
from causal_ai_sdk.config import Config
from causal_ai_sdk.contracts.requests import CDLingamRunRequestContract
from causal_ai_sdk.models.cd import UploadedData
from causal_ai_sdk.services.lingam_service import LingamService


def _make_lingam_service(mock_http=None):
    if mock_http is None:
        mock_http = AsyncMock()
    config = Config(api_key="test-key", base_url="https://api.test")
    return LingamService(config, mock_http), mock_http


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def test_run_lingam_calls_post_with_correct_path_and_body():
    """run_lingam POSTs to /cd/lingam/run/{uuid} with task_id, s3_key, threshold/params."""
    service, mock_http = _make_lingam_service()
    mock_http.request = AsyncMock(return_value={"uuid": "s1", "task_id": "t1", "status": "queued"})

    uploaded = UploadedData(
        task_id="cd-task-1",
        session_uuid="s1",
        s3_key="cd/s1/cd-task-1/input",
    )

    result = _run(service.run_lingam(uploaded, threshold=0.01, params={"gamma": 1.0}))

    assert isinstance(result, dict)
    assert result["task_id"] == "t1"
    assert result["status"] == "queued"
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "POST"
    assert "/cd/lingam/run/s1" in call_kw["url"]
    assert call_kw["json_data"]["task_id"] == "cd-task-1"
    assert call_kw["json_data"]["s3_key"] == "cd/s1/cd-task-1/input"
    assert call_kw["json_data"]["threshold"] == 0.01
    assert call_kw["json_data"]["params"] == {"gamma": 1.0}


def test_cdlingam_run_request_contract_validates():
    """Verify CDLingamRunRequestContract accepts valid payload and optional threshold/params."""
    payload = {
        "task_id": "t1",
        "s3_key": "cd/s1/t1/input",
    }
    model = CDLingamRunRequestContract.model_validate(payload)
    assert model.task_id == "t1"
    assert model.s3_key == "cd/s1/t1/input"
    assert model.threshold is None
    assert model.params is None

    payload_with_optional = {
        "task_id": "t2",
        "s3_key": "cd/s2/t2/input",
        "threshold": 0.05,
        "params": {"gamma": 1.0, "delta": 2.0},
    }
    model2 = CDLingamRunRequestContract.model_validate(payload_with_optional)
    assert model2.threshold == 0.05
    assert model2.params == {"gamma": 1.0, "delta": 2.0}


def test_cdlingam_run_request_contract_rejects_invalid_threshold():
    """Verify CDLingamRunRequestContract rejects threshold outside (0, 1]."""
    with pytest.raises(ValueError):
        CDLingamRunRequestContract.model_validate(
            {"task_id": "t1", "s3_key": "k", "threshold": 1.5}
        )
    with pytest.raises(ValueError):
        CDLingamRunRequestContract.model_validate(
            {"task_id": "t1", "s3_key": "k", "threshold": 0.0}
        )
