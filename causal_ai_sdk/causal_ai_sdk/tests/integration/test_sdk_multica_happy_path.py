"""SDK integration: MultiCa happy path using CausalAIClient.

Mirrors MVP tests/causal_discovery/integration/test_multica_happy_path.py.
Requires deployed backend. Run with pytest -n auto -m integration from MVP.
"""

from __future__ import annotations

import httpx
import pytest
from causal_ai_sdk import CausalAIClient
from causal_ai_sdk.config import Config


@pytest.mark.asyncio
async def test_sdk_multica_happy_path(
    sdk_base_url, sdk_test_data_dir, sdk_request_timeout, sdk_polling_timeout, make_api_key
):
    """Run MultiCa happy path via SDK: init, upload, match, run, wait, result, cleanup.

    Args:
        sdk_base_url (str): API base URL for client.
        sdk_test_data_dir (Path): Path to test data.
        sdk_request_timeout (int): HTTP request timeout (from fixture).
        sdk_polling_timeout (int): Polling/wait timeout (from fixture).
        make_api_key (Callable): Fixture to create API key.
    """
    api_key = make_api_key("SDKMultiCaIntegration", token_balance=10)
    multica_file = sdk_test_data_dir / "cd" / "multica_datasets.json"
    if not multica_file.exists():
        pytest.skip(f"Test data not found: {multica_file}")

    config = Config(api_key=api_key, base_url=sdk_base_url, timeout=sdk_request_timeout)
    async with CausalAIClient(api_key=api_key, base_url=sdk_base_url, config=config) as client:
        session = await client.kg.init_session()
        session_uuid = session["uuid"]
        run_task_id = None
        matching_task_id = None
        try:
            uploaded_data = await client.cd.upload_data_for_multica(session_uuid, multica_file)
            matching_task = await client.cd.start_multica_matching(uploaded_data)
            matching_task_id = matching_task["matching_task_id"]

            await client.cd.wait_for_matching(
                session_uuid,
                mode="multica",
                matching_task_id=matching_task_id,
                timeout=sdk_polling_timeout,
                interval=5,
            )
            task = await client.cd.run_multica(
                uploaded_data,
                threshold=0.05,
                matching_task_id=matching_task_id,
            )
            run_task_id = task["task_id"]

            await client.cd.wait_for_task(
                run_task_id, session_uuid=session_uuid, timeout=sdk_polling_timeout, interval=5
            )
            task_result = await client.cd.get_task_result(session_uuid, run_task_id)
            assert task_result.get("result_url"), "Missing result_url in task result"

            async with httpx.AsyncClient() as http_client:
                resp = await http_client.get(task_result["result_url"], timeout=30.0)
                resp.raise_for_status()
                result = resp.json()
            assert result.get("status") == "succeeded"
            assert result.get("mode") == "multica"
            assert result.get("num_datasets") == 2
            assert "feature_names" in result
            assert "adjacency_matrices" in result
            assert "matching_used" in result
        finally:
            if run_task_id and session_uuid:
                try:
                    await client.cd.delete_task(session_uuid, run_task_id)
                except Exception:
                    pass
            if matching_task_id and session_uuid:
                try:
                    await client.cd.delete_matching(session_uuid, "multica", matching_task_id)
                except Exception:
                    pass
            if session_uuid:
                try:
                    await client.kg.delete_session(session_uuid)
                except Exception:
                    pass
