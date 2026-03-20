"""Unit tests for SDK Decision Analysis (DA) service."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock

import pytest
from causal_ai_sdk.config import Config
from causal_ai_sdk.services.da_service import DAService


def _make_da_service(mock_http=None):
    if mock_http is None:
        mock_http = AsyncMock()
    config = Config(api_key="test-key", base_url="https://api.test")
    return DAService(config, mock_http), mock_http


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run async coroutine in sync test.

    Args:
        coro (Coroutine[Any, Any, Any]): Coroutine to run to completion.

    Returns:
        Any: The return value of the coroutine.
    """
    return asyncio.run(coro)


def test_run_explain_calls_post_with_correct_path_and_body():
    """run_explain sends POST to /da/explain/{uuid} with validated body."""
    service, mock_http = _make_da_service()
    mock_http.request = AsyncMock(return_value={"uuid": "s1", "task_id": "t1", "status": "queued"})

    result = _run(
        service.run_explain(
            session_uuid="s1",
            cd_result_reference={"session_uuid": "s1", "task_id": "cd-1"},
            current_observation={"x": 0.5, "y": 0.3},
            targets=[{"col": "y", "sense": ">", "threshold": 0.8}],
        )
    )

    assert isinstance(result, dict)
    assert result["uuid"] == "s1"
    assert result["task_id"] == "t1"
    assert result["status"] == "queued"
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "POST"
    assert "/da/explain/s1" in call_kw["url"]
    assert call_kw["json_data"]["cd_result_reference"] == {"session_uuid": "s1", "task_id": "cd-1"}
    assert call_kw["json_data"]["current_observation"] == {"x": 0.5, "y": 0.3}
    assert call_kw["json_data"]["targets"] == [{"col": "y", "sense": ">", "threshold": 0.8}]


def test_run_enumerate_calls_post_with_correct_path_and_body():
    """run_enumerate sends POST to /da/enumerate/{uuid} with validated body."""
    service, mock_http = _make_da_service()
    mock_http.request = AsyncMock(return_value={"uuid": "s1", "task_id": "t1", "status": "queued"})

    result = _run(
        service.run_enumerate(
            session_uuid="s2",
            cd_result_reference={"session_uuid": "s2", "task_id": "cd-2"},
            current_observation={"a": 0.1},
            targets=[{"col": "a", "sense": "<", "threshold": 0.5}],
        )
    )

    assert isinstance(result, dict)
    assert result["task_id"] == "t1"
    mock_http.request.assert_awaited_once()
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "POST"
    assert "/da/enumerate/s2" in call_kw["url"]
    assert call_kw["json_data"]["cd_result_reference"] == {"session_uuid": "s2", "task_id": "cd-2"}
    assert call_kw["json_data"]["current_observation"] == {"a": 0.1}
    assert call_kw["json_data"]["targets"] == [{"col": "a", "sense": "<", "threshold": 0.5}]


def test_get_task_status_calls_get_with_task_id_param():
    """get_task_status sends GET to /da/status/{uuid}?task_id=..."""
    service, mock_http = _make_da_service()
    mock_http.request = AsyncMock(
        return_value={"uuid": "s1", "task_id": "t1", "status": "succeeded", "error": None}
    )

    result = _run(service.get_task_status(session_uuid="s1", task_id="t1"))

    assert isinstance(result, dict)
    assert result["status"] == "succeeded"
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "GET"
    assert "/da/status/s1" in call_kw["url"]
    assert call_kw["params"] == {"task_id": "t1"}


def test_get_task_result_calls_get_with_task_id_param():
    """get_task_result sends GET to /da/result/{uuid}?task_id=..."""
    service, mock_http = _make_da_service()
    mock_http.request = AsyncMock(
        return_value={
            "uuid": "s1",
            "task_id": "t1",
            "status": "succeeded",
            "result_url": "https://s3.example/presigned",
            "expires_in": 3600,
        }
    )

    result = _run(service.get_task_result(session_uuid="s1", task_id="t1"))

    assert isinstance(result, dict)
    assert result["result_url"] == "https://s3.example/presigned"
    assert result["expires_in"] == 3600
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["params"] == {"task_id": "t1"}
    assert "/da/result/s1" in call_kw["url"]


def test_delete_task_calls_delete_with_task_id_param():
    """delete_task sends DELETE to /da/task/{uuid}?task_id=..."""
    service, mock_http = _make_da_service()
    mock_http.request = AsyncMock(return_value={"uuid": "s1", "task_id": "t1", "status": "deleted"})

    _run(service.delete_task(session_uuid="s1", task_id="t1"))

    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["method"] == "DELETE"
    assert "/da/task/s1" in call_kw["url"]
    assert call_kw["params"] == {"task_id": "t1"}


def test_run_explain_validates_threshold_tuple_lb_le_ub():
    """run_explain raises when (lb, ub) threshold has lb > ub."""
    service, _ = _make_da_service()

    with pytest.raises(ValueError, match="lower bound <= upper bound"):
        _run(
            service.run_explain(
                session_uuid="s1",
                cd_result_reference={"session_uuid": "s1", "task_id": "cd-1"},
                current_observation={"x": 0.5},
                targets=[{"col": "x", "sense": "in", "threshold": [0.8, 0.2]}],
            )
        )


def test_run_explain_validates_sense_in_requires_range():
    """run_explain raises when sense is 'in' but threshold is scalar."""
    service, _ = _make_da_service()

    with pytest.raises(ValueError, match="sense 'in'"):
        _run(
            service.run_explain(
                session_uuid="s1",
                cd_result_reference={"session_uuid": "s1", "task_id": "cd-1"},
                current_observation={"x": 0.5},
                targets=[{"col": "x", "sense": "in", "threshold": 0.8}],
            )
        )


def test_run_explain_validates_sense_gt_requires_scalar():
    """run_explain raises when sense is '>' but threshold is range."""
    service, _ = _make_da_service()

    with pytest.raises(ValueError, match="sense '>' or '<'"):
        _run(
            service.run_explain(
                session_uuid="s1",
                cd_result_reference={"session_uuid": "s1", "task_id": "cd-1"},
                current_observation={"x": 0.5},
                targets=[{"col": "x", "sense": ">", "threshold": [0.0, 1.0]}],
            )
        )


def test_run_explain_allows_current_observation_to_omit_keys():
    """run_explain accepts current_observation with omitted keys (imputed at runtime)."""
    service, mock_http = _make_da_service()
    mock_http.request = AsyncMock(return_value={"uuid": "s1", "task_id": "t1", "status": "queued"})

    # target is "b"; current_observation only has "a" (b omitted)
    result = _run(
        service.run_explain(
            session_uuid="s1",
            cd_result_reference={"session_uuid": "s1", "task_id": "cd-1"},
            current_observation={"a": 0.5},
            targets=[{"col": "b", "sense": ">", "threshold": 0.8}],
        )
    )

    assert isinstance(result, dict)
    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["json_data"]["current_observation"] == {"a": 0.5}
    assert call_kw["json_data"]["targets"] == [{"col": "b", "sense": ">", "threshold": 0.8}]


def test_run_explain_accepts_per_target_objects():
    """run_explain accepts targets=[{col,sense,threshold}] shape."""
    service, mock_http = _make_da_service()
    mock_http.request = AsyncMock(return_value={"uuid": "s1", "task_id": "t1", "status": "queued"})

    _run(
        service.run_explain(
            session_uuid="s1",
            cd_result_reference={"session_uuid": "s1", "task_id": "cd-1"},
            current_observation={"debt": 0.4},
            targets=[{"col": "debt", "sense": ">", "threshold": 0.8}],
            params={"alpha": 1.0, "time_limit": 60.0},
        )
    )

    call_kw = mock_http.request.await_args.kwargs
    assert call_kw["json_data"]["targets"][0]["sense"] == ">"


def test_run_explain_rejects_multi_target_for_now():
    """run_explain rejects multi-target payload while feature is gated."""
    service, _ = _make_da_service()
    with pytest.raises(ValueError, match="targets length must be 1"):
        _run(
            service.run_explain(
                session_uuid="s1",
                cd_result_reference={"session_uuid": "s1", "task_id": "cd-1"},
                current_observation={"debt": 0.4, "age": 22.0},
                targets=[
                    {"col": "debt", "sense": ">", "threshold": 0.8},
                    {"col": "age", "sense": "in", "threshold": [20.0, 30.0]},
                ],
                params={"alpha": 1.0, "time_limit": 60.0},
            )
        )
