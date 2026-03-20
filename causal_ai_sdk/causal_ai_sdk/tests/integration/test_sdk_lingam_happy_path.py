"""SDK integration: LiNGAM happy path using CausalAIClient.

Requires deployed backend (e.g. LocalStack). Marked as integration; run with
pytest -n auto -m integration (includes causal_ai_sdk) from MVP.
"""

from __future__ import annotations

import pytest
from causal_ai_sdk import CausalAIClient
from causal_ai_sdk.config import Config


@pytest.mark.asyncio
async def test_sdk_lingam_happy_path(
    sdk_base_url, sdk_test_data_dir, sdk_request_timeout, sdk_polling_timeout, make_api_key
):
    """Run LiNGAM happy path via SDK: init, upload, run, wait, result, cleanup.

    Args:
        sdk_base_url (str): API base URL for client.
        sdk_test_data_dir (Path): Path to test data.
        sdk_request_timeout (int): HTTP request timeout (from fixture).
        sdk_polling_timeout (int): Polling/wait timeout (from fixture).
        make_api_key (Callable): Fixture to create API key.
    """
    api_key = make_api_key("SDKLingamIntegration", token_balance=10)
    data_file = sdk_test_data_dir / "cd" / "trackr_target_data.json"
    if not data_file.exists():
        pytest.skip(f"Test data not found: {data_file}")

    config = Config(api_key=api_key, base_url=sdk_base_url, timeout=sdk_request_timeout)
    async with CausalAIClient(api_key=api_key, base_url=sdk_base_url, config=config) as client:
        session = await client.kg.init_session()
        session_uuid = session["uuid"]
        run_task_id = None
        try:
            uploaded_data = await client.cd.upload_data_for_lingam(session_uuid, data_file)
            task = await client.cd.run_lingam(uploaded_data, threshold=0.01)
            run_task_id = task["task_id"]

            await client.cd.wait_for_task(
                run_task_id, session_uuid=session_uuid, timeout=sdk_polling_timeout, interval=5
            )
            task_result = await client.cd.get_task_result(session_uuid, run_task_id)
            assert task_result.get("result_url"), "Missing result_url in task result"
        finally:
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
