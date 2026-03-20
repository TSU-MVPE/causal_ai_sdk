"""SDK integration: DA explain happy path after LiNGAM using CausalAIClient.

Mirrors MVP tests/decision_analysis/integration/test_da_explain_happy_path_after_lingam.py.
Requires deployed backend. Run with pytest -n auto -m integration from MVP.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import httpx
import pytest
from causal_ai_sdk import CausalAIClient
from causal_ai_sdk.config import Config


def _build_one_hot_set(cd_result: Dict[str, Any]) -> set:
    """Set of one-hot feature names from CD result feature_metadata.

    Args:
        cd_result (Dict[str, Any]): CD result dict with feature_names and feature_metadata.

    Returns:
        Set of one-hot encoded feature names.
    """
    names = cd_result.get("feature_names") or []
    categories = (cd_result.get("feature_metadata") or {}).get("feature_categories") or []
    one_hot = {n for n in names if "=" in n}
    for group in categories:
        if isinstance(group, list):
            for idx in group:
                if isinstance(idx, int) and 0 <= idx < len(names):
                    one_hot.add(names[idx])
    return one_hot


def _build_current_observation_and_targets(
    cd_result: Dict[str, Any],
) -> Tuple[Dict[str, float], List[str]]:
    """Build current_observation (first row) and targets (first non-one-hot) from CD result.

    Args:
        cd_result (Dict[str, Any]): CD result with feature_names, training_data, feature_metadata.

    Returns:
        Tuple of (current_observation dict, list of target feature names).
    """
    names = cd_result.get("feature_names") or []
    rows = (cd_result.get("training_data") or {}).get("data") or []
    if not names or not rows:
        return {}, []
    first_row = rows[0]
    if not isinstance(first_row, list) or len(first_row) != len(names):
        return {}, []
    one_hot = _build_one_hot_set(cd_result)
    obs: Dict[str, float] = {}
    for name, value in zip(names, first_row):
        if name in one_hot:
            obs[name] = int(value) if value in (0, 1, 0.0, 1.0, False, True) else 0
        else:
            obs[name] = 0.0 if (isinstance(value, float) and math.isnan(value)) else value
    candidates = [n for n in names if n not in one_hot]
    targets = [candidates[0]] if candidates else []
    return obs, targets


@pytest.mark.asyncio
async def test_sdk_da_explain_happy_path(
    sdk_base_url, sdk_test_data_dir, sdk_request_timeout, sdk_polling_timeout, make_api_key
):
    """Run LiNGAM to completion then DA explain: init, CD run, DA explain, wait, result, cleanup.

    Args:
        sdk_base_url (str): API base URL for client.
        sdk_test_data_dir (Path): Path to test data.
        sdk_request_timeout (int): HTTP request timeout (from fixture).
        sdk_polling_timeout (int): Polling/wait timeout (from fixture).
        make_api_key (Callable): Fixture to create API key.
    """
    api_key = make_api_key("SDKDAExplainIntegration", token_balance=10)
    data_file = sdk_test_data_dir / "cd" / "trackr_target_data.json"
    if not data_file.exists():
        pytest.skip(f"Test data not found: {data_file}")

    config = Config(api_key=api_key, base_url=sdk_base_url, timeout=sdk_request_timeout)
    async with CausalAIClient(api_key=api_key, base_url=sdk_base_url, config=config) as client:
        session = await client.kg.init_session()
        session_uuid = session["uuid"]
        run_task_id = None
        da_task_id = None
        try:
            uploaded_data = await client.cd.upload_data_for_lingam(session_uuid, data_file)
            task = await client.cd.run_lingam(uploaded_data, threshold=0.05)
            run_task_id = task["task_id"]
            await client.cd.wait_for_task(
                run_task_id, session_uuid=session_uuid, timeout=sdk_polling_timeout, interval=5
            )
            task_result = await client.cd.get_task_result(session_uuid, run_task_id)
            assert task_result.get("result_url"), "Missing result_url in CD task result"
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.get(task_result["result_url"], timeout=30.0)
                resp.raise_for_status()
                cd_result = resp.json()
            assert cd_result.get("status") == "succeeded" and cd_result.get("mode") == "lingam"

            current_obs, targets = _build_current_observation_and_targets(cd_result)
            if not targets:
                current_obs = {"x": 0.5}
                targets = ["x"]
            target_col = targets[0]
            cd_result_reference = {
                "session_uuid": session_uuid,
                "task_id": run_task_id,
            }

            da_task = await client.da.run_explain(
                session_uuid=session_uuid,
                cd_result_reference=cd_result_reference,
                current_observation=current_obs,
                targets=[{"col": target_col, "sense": ">", "threshold": 0.8}],
            )
            da_task_id = da_task["task_id"]
            assert da_task.get("status") == "queued"

            await client.da.wait_for_task(
                session_uuid, da_task_id, timeout=sdk_polling_timeout, interval=5
            )
            da_result_resp = await client.da.get_task_result(session_uuid, da_task_id)
            assert da_result_resp.get("result_url"), "Missing result_url in DA task result"
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.get(da_result_resp["result_url"], timeout=30.0)
                resp.raise_for_status()
                da_result = resp.json()
            assert da_result.get("status") == "succeeded"
            assert "result" in da_result
        finally:
            if da_task_id and session_uuid:
                try:
                    await client.da.delete_task(session_uuid, da_task_id)
                except Exception:
                    pass
            if run_task_id and session_uuid:
                try:
                    await client.cd.delete_task(session_uuid, run_task_id)
                except Exception:
                    pass
            if session_uuid:
                try:
                    await client.kg.delete_session(session_uuid)
                except Exception:
                    pass
