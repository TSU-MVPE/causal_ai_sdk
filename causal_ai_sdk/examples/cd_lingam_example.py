"""Example usage of CD Service - LiNGAM mode from Causal AI SDK.

This example demonstrates the LiNGAM causal discovery workflow (no matching):
1. Create a session
2. Upload a single dataset to S3
3. Submit LiNGAM task
4. Wait for task to complete (polling handled by SDK)
5. Get and validate results
6. Cleanup (delete task, delete session)
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import httpx
from causal_ai_sdk import CausalAIClient
from causal_ai_sdk.models.cd import UploadedData
from helpers import (
    _delete_api_key,
    _get_api_key,
    _resolve_api_base_url,
    get_sdk_test_data_dir,
)


async def _run_lingam_workflow(
    client: CausalAIClient,
    session_uuid: str,
    uploaded_data: UploadedData,
) -> tuple[bool, Optional[str]]:
    """Run the complete LiNGAM workflow (upload already done).

    Args:
        client (CausalAIClient): Causal AI client instance
        session_uuid (str): Session UUID
        uploaded_data (UploadedData): From upload_data_for_lingam

    Returns:
        tuple[bool, Optional[str]]: (success, run_task_id for cleanup or None)
    """
    run_task_id: Optional[str] = None
    try:
        # Step 3: Submit LiNGAM task
        print("  Step 3: Submit LiNGAM Task")
        task = await client.cd.run_lingam(uploaded_data, threshold=0.01)
        run_task_id = task["task_id"]
        print(f"  [OK] Task submitted: {run_task_id}")

        # Step 4: Wait for task to complete (use run task_id, not upload task_id)
        print("  Step 4: Wait for Task to Complete")
        await client.cd.wait_for_task(
            run_task_id, session_uuid=session_uuid, timeout=600, interval=5
        )
        print("  [OK] Task completed successfully")

        # Step 5: Get and validate results
        print("  Step 5: Retrieve Results")
        task_result = await client.cd.get_task_result(session_uuid, run_task_id)
        result_url = task_result.get("result_url")
        if not result_url:
            print("  [ERROR] No result URL in response")
            return False, run_task_id

        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(result_url)
            resp.raise_for_status()
            result_data = resp.json()

        required = [
            "mode",
            "feature_names",
            "adjacency_matrices",
            "training_data",
            "feature_metadata",
            "status",
        ]
        missing = [k for k in required if k not in result_data]
        if missing:
            print(f"  [ERROR] Result structure incomplete (missing: {missing})")
            return False, run_task_id
        if result_data.get("mode") != "lingam":
            print(f"  [ERROR] Expected mode=lingam, got: {result_data.get('mode')}")
            return False, run_task_id
        if result_data.get("status") != "succeeded":
            print(f"  [ERROR] Expected status=succeeded, got: {result_data.get('status')}")
            return False, run_task_id

        print("  [OK] Results retrieved and validated")
        print(f"    Mode: {result_data.get('mode')}")
        print(f"    Variables: {len(result_data.get('feature_names', []))}")

        return True, run_task_id

    except Exception as e:
        print(f"  [ERROR] Workflow failed: {e}")
        return False, run_task_id


async def test_lingam_json(
    client: CausalAIClient, test_data_dir: Path
) -> tuple[bool, Optional[str]]:
    """Test LiNGAM workflow with a single JSON dataset.

    Args:
        client (CausalAIClient): Causal AI client instance
        test_data_dir (Path): Directory containing test data (e.g. scripts/test_data/cd)

    Returns:
        tuple[bool, Optional[str]]: (success, session_uuid) or (False, None) if file not found
    """
    # Use same target data as e2e LiNGAM (single dataset JSON)
    data_file = test_data_dir / "trackr_target_data.json"
    if not data_file.exists():
        print("  Test file not found (trackr_target_data.json), skipping...")
        return False, None

    print("\n" + "=" * 60)
    print("Testing LiNGAM Causal Discovery")
    print("=" * 60)

    session_uuid = None
    run_task_id: Optional[str] = None
    try:
        # Step 1: Create session
        print("  Step 1: Create Session")
        session = await client.kg.init_session()
        session_uuid = session["uuid"]
        print(f"  [OK] Session created: {session_uuid}")

        # Step 2: Upload data (single file, no matching)
        print("  Step 2: Upload Data")
        uploaded_data = await client.cd.upload_data_for_lingam(session_uuid, data_file)
        print(f"  [OK] Data uploaded: task_id={uploaded_data.task_id}")

        # Steps 3: Run LiNGAM, wait, get result (returns run task_id for cleanup)
        success, run_task_id = await _run_lingam_workflow(client, session_uuid, uploaded_data)

        return success, session_uuid

    except Exception as e:
        print(f"  [ERROR] Test failed: {e}")
        return False, None

    finally:
        # Cleanup: delete run task (use run task_id, not upload task_id) then session
        if session_uuid:
            print("  Cleanup")
            if run_task_id:
                try:
                    await client.cd.delete_task(session_uuid, run_task_id)
                    print("  [OK] CD task deleted")
                except Exception as e:
                    print(f"  [ERROR] (CD delete: {e})")
            try:
                await client.kg.delete_session(session_uuid)
                print("  [OK] Session deleted")
            except Exception as e:
                print(f"  [ERROR] (Session delete: {e})")


async def main() -> None:
    """Run LiNGAM example."""
    base_url = _resolve_api_base_url()
    if not base_url:
        print(
            "\n[ERROR] Could not determine API Gateway URL.\n"
            "Set CAUSAL_AI_BASE_URL or ensure Terraform outputs are available."
        )
        sys.exit(1)

    api_key, temp_company_name = _get_api_key()
    if not api_key:
        print("\n[ERROR] No API key. Set CAUSAL_AI_API_KEY or use cai-keymgr.")
        sys.exit(1)

    print("=" * 60)
    print("LiNGAM Causal Discovery Example")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print(f"API Key: {api_key[:15]}..." if len(api_key) > 15 else f"API Key: {api_key}")
    print("=" * 60)

    test_data_dir = get_sdk_test_data_dir() / "cd"

    try:
        async with CausalAIClient(api_key=api_key, base_url=base_url) as client:
            success, _ = await test_lingam_json(client, test_data_dir)
        print("\n" + "=" * 60)
        print("Summary: " + ("[OK] PASSED" if success else "[ERROR] FAILED/SKIPPED"))
        print("=" * 60)
    finally:
        if temp_company_name:
            _delete_api_key(temp_company_name)


if __name__ == "__main__":
    asyncio.run(main())
