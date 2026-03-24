"""Example: LiNGAM causal discovery followed by Decision Analysis (explain).

This example runs the full flow:
1. Create session
2. Upload data and run LiNGAM CD
3. Wait for CD result and download it
4. Build current_observation and targets from CD result (first row, first non-one-hot target)
5. Submit DA explain task using CD result reference
6. Wait for DA task and fetch result
7. Cleanup
"""

import asyncio
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from causal_ai_sdk import CausalAIClient
from helpers import (
    get_api_key_from_env,
    get_base_url_from_env,
    get_sdk_test_data_dir,
)


def _build_one_hot_set(cd_result: Dict[str, Any]) -> set:
    """Build set of one-hot feature names from CD result.

    Args:
        cd_result (Dict[str, Any]): CD result dict with feature_names and feature_metadata.

    Returns:
        Set of feature names that are one-hot encoded.
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


def _is_nan(v: Any) -> bool:
    """Return True if value is NaN.

    Args:
        v (Any): Value to check.

    Returns:
        True if v is a float and NaN, else False.
    """
    return isinstance(v, float) and math.isnan(v)


def _build_current_observation_and_targets(
    cd_result: Dict[str, Any],
) -> Tuple[Dict[str, float], List[str]]:
    """Build current_observation (first row) and targets (first non-one-hot) from CD result.

    One-hot: 0/1 or 0 if NaN; numeric: 0.0 if NaN else value.
    Targets: first non-one-hot feature name.

    Args:
        cd_result (Dict[str, Any]): CD result with feature_names, training_data,
            feature_metadata.

    Returns:
        Tuple of (current_observation dict, list of target names).

    Raises:
        ValueError: If feature_names/training_data missing or no non-one-hot target.
    """
    names = cd_result.get("feature_names") or []
    rows = (cd_result.get("training_data") or {}).get("data") or []
    if not names or not rows:
        raise ValueError("CD result missing feature_names or training_data.data")
    first_row = rows[0]
    if len(first_row) != len(names):
        raise ValueError("First row length does not match feature_names")

    one_hot = _build_one_hot_set(cd_result)
    obs: Dict[str, float] = {}
    for name, value in zip(names, first_row):
        if name in one_hot:
            obs[name] = int(value) if value in (0, 1, 0.0, 1.0, False, True) else 0
        else:
            obs[name] = 0.0 if _is_nan(value) else value

    candidates = [n for n in names if n not in one_hot]
    targets = [candidates[0]] if candidates else []
    if not targets:
        raise ValueError("No non-one-hot feature for target")
    return obs, targets


async def run_lingam_da_workflow(
    client: CausalAIClient,
    test_data_dir: Path,
) -> bool:
    """Run LiNGAM then DA explain and return True if both succeed.

    Args:
        client (CausalAIClient): Causal AI client instance.
        test_data_dir (Path): Path to directory containing trackr_target_data.json.

    Returns:
        True if both LiNGAM and DA workflows succeed, else False.
    """
    data_file = test_data_dir / "trackr_target_data.json"
    if not data_file.exists():
        print("  Test file not found (trackr_target_data.json), skipping...")
        return False

    session_uuid: Optional[str] = None
    cd_task_id: Optional[str] = None
    da_task_id: Optional[str] = None

    try:
        # --- Session ---
        print("  Step 1: Create Session")
        session = await client.kg.init_session()
        session_uuid = session["uuid"]
        print(f"  [OK] Session: {session_uuid}")

        # --- LiNGAM CD ---
        print("  Step 2: Upload Data and Run LiNGAM")
        uploaded_data = await client.cd.upload_data_for_lingam(session_uuid, data_file)
        print("  [OK] Data Uploaded")

        print("  Step 3: Submit LiNGAM Task")
        task = await client.cd.run_lingam(uploaded_data, threshold=0.01)
        cd_task_id = task["task_id"]
        print(f"  [OK] LiNGAM task submitted: {cd_task_id}")

        print("  Step 4: Wait for LiNGAM to Complete")
        await client.cd.wait_for_task(
            cd_task_id, session_uuid=session_uuid, timeout=600, interval=5
        )
        print("  [OK] LiNGAM completed")

        print("  Step 5: Fetch CD Result")
        task_result = await client.cd.get_task_result(session_uuid, cd_task_id)
        if not task_result.get("result_url"):
            print("  [ERROR] No CD result URL")
            return False
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(task_result.get("result_url"))
            resp.raise_for_status()
            cd_result = resp.json()
        if cd_result.get("status") != "succeeded" or cd_result.get("mode") != "lingam":
            print("  [ERROR] CD result not succeeded or not lingam mode")
            return False
        print("  [OK] CD result retrieved")

        # --- Build DA input from CD result ---
        print("  Step 6: Build DA Input from CD Result")
        current_observation, targets = _build_current_observation_and_targets(cd_result)
        cd_result_reference = {"session_uuid": session_uuid, "task_id": cd_task_id}
        da_targets = [{"col": targets[0], "sense": ">", "threshold": 0.8}]
        print(f"  [OK] current_observation keys: {len(current_observation)}, targets: {targets}")

        # --- DA explain ---
        print("  Step 7: Submit DA Explain Task")
        da_task = await client.da.run_explain(
            session_uuid=session_uuid,
            cd_result_reference=cd_result_reference,
            current_observation=current_observation,
            targets=da_targets,
        )
        da_task_id = da_task["task_id"]
        print(f"  [OK] DA task submitted: {da_task_id}")

        print("  Step 8: Wait for DA Task to Complete")
        await client.da.wait_for_task(session_uuid, da_task_id, timeout=300, interval=5)
        print("  [OK] DA task succeeded")

        print("  Step 9: Fetch DA Result")
        da_result_resp = await client.da.get_task_result(session_uuid, da_task_id)
        if not da_result_resp.get("result_url"):
            print("  [ERROR] No DA result URL")
            return False
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(da_result_resp.get("result_url"))
            resp.raise_for_status()
            da_result = resp.json()
        if da_result.get("status") != "succeeded":
            print("  [ERROR] DA result status not succeeded")
            return False
        if "result" not in da_result:
            print("  [ERROR] DA result missing 'result' key")
            return False
        print("  [OK] DA result retrieved")
        print(f"    is_solved: {(da_result.get('result') or {}).get('is_solved')}")

        return True

    except Exception as e:
        print(f"  [ERROR] Workflow failed: {e}")
        return False

    finally:
        # Cleanup
        print("  Step 10: Cleanup")
        if session_uuid:
            if da_task_id:
                try:
                    await client.da.delete_task(session_uuid, da_task_id)
                    print("  [OK] DA task deleted")
                except Exception as e:
                    print(f"  [ERROR] (DA delete: {e})")
            if cd_task_id:
                try:
                    await client.cd.delete_task(session_uuid, cd_task_id)
                    print("  [OK] CD task deleted")
                except Exception as e:
                    print(f"  [ERROR] (CD delete: {e})")
            try:
                await client.kg.delete_session(session_uuid)
                print("  [OK] Session deleted")
            except Exception as e:
                print(f"  [ERROR] (Session delete: {e})")


async def main() -> None:
    """Run LiNGAM + DA example."""
    base_url = get_base_url_from_env()
    if not base_url:
        print("\n[ERROR] No API base URL. Set CAUSAL_AI_BASE_URL.")
        sys.exit(1)

    api_key = get_api_key_from_env()
    if not api_key:
        print("\n[ERROR] No API key. Set CAUSAL_AI_API_KEY.")
        sys.exit(1)

    print("=" * 60)
    print("LiNGAM + DA Example")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print("=" * 60)

    test_data_dir = get_sdk_test_data_dir() / "cd"

    async with CausalAIClient(api_key=api_key, base_url=base_url) as client:
        success = await run_lingam_da_workflow(client, test_data_dir)
    print("\n" + "=" * 60)
    print("Summary: " + ("[OK] PASSED" if success else "[ERROR] FAILED/SKIPPED"))
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
