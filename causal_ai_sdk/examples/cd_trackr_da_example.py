"""Example: TraCKR causal discovery followed by Decision Analysis (explain).

This example runs the full flow:
1. Create session
2. Upload and register source KG, upload CD target data
3. TraCKR matching
4. Submit TraCKR task and wait for CD result
5. Build current_observation and targets from CD result
6. Submit DA explain task using CD result reference
7. Wait for DA task and fetch result
8. Cleanup
"""

import asyncio
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from causal_ai_sdk import CausalAIClient
from helpers import (
    _delete_api_key,
    _get_api_key,
    _resolve_api_base_url,
    get_sdk_test_data_dir,
)


def _build_one_hot_set(cd_result: Dict[str, Any]) -> set:
    """Collect one-hot encoded feature names from CD result feature_metadata.

    Args:
        cd_result (Dict[str, Any]): CD result dict with feature_names and feature_metadata.

    Returns:
        Set of one-hot encoded feature names.
    """
    names = cd_result.get("feature_names") or []
    meta = cd_result.get("feature_metadata") or {}
    categories = meta.get("feature_categories") or []
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
    """Build DA current_observation (first row) and targets (first non-one-hot) from CD result.

    One-hot: 0/1 or 0 if invalid/NaN; numeric: 0.0 if NaN else value.
    Aligned with cd_lingam_da_example.py.

    Args:
        cd_result (Dict[str, Any]): CD result with training_data, feature_names,
            feature_metadata.

    Returns:
        Tuple of (current_observation dict, targets list).
    """
    names = cd_result.get("feature_names") or []
    rows = (cd_result.get("training_data") or {}).get("data") or []
    if not names or not rows:
        return {}, []
    first_row = rows[0]
    if not isinstance(first_row, list) or len(first_row) != len(names):
        return {}, []
    one_hot = _build_one_hot_set(cd_result)
    current_observation: Dict[str, float] = {}
    for name, value in zip(names, first_row):
        if name in one_hot:
            current_observation[name] = int(value) if value in (0, 1, 0.0, 1.0, False, True) else 0
        else:
            current_observation[name] = 0.0 if _is_nan(value) else value

    candidates = [n for n in names if n not in one_hot]
    targets = [candidates[0]] if candidates else []

    return current_observation, targets


async def run_trackr_da_workflow(client: CausalAIClient, test_data_dir: Path) -> bool:
    """Run TraCKR CD then DA explain; cleanup in finally.

    Args:
        client (CausalAIClient): Causal AI client instance.
        test_data_dir (Path): Path to test data directory (e.g. scripts/test_data).

    Returns:
        True if workflow completed successfully, False otherwise.
    """
    source_kg_path = test_data_dir / "kg" / "source_graph.json"
    target_data_path = test_data_dir / "cd" / "trackr_target_data.json"
    if not source_kg_path.exists():
        print(f"  [ERROR] Source KG file not found: {source_kg_path}")
        return False
    if not target_data_path.exists():
        print(f"  [ERROR] Target data file not found: {target_data_path}")
        return False

    session_uuid = None
    kg_id = None
    uploaded_data = None
    matching_task_id = None
    cd_task_id = None
    da_task_id = None

    try:
        # Step 1: Create Session
        print("  Step 1: Create Session")
        session = await client.kg.init_session()
        session_uuid = session["uuid"]
        print(f"  [OK] Session created: {session_uuid}")

        # Step 2: Upload and Register Source KG
        print("  Step 2: Upload and Register Source KG")
        kg = await client.kg.upload_kg_from_file(
            session_uuid=session_uuid, file_path=source_kg_path
        )
        kg_id = kg["id"]
        print(f"  [OK] KG registered: {kg_id}")

        # Step 3: Upload CD Target Data
        print("  Step 3: Upload CD Target Data")
        uploaded_data = await client.cd.upload_data_for_trackr(session_uuid, target_data_path)
        print("  [OK] CD target uploaded")

        # Step 4: TraCKR Matching
        print("  Step 4: TraCKR Matching")
        matching_task = await client.cd.start_trackr_matching(
            uploaded_data=uploaded_data, source_kg_id=kg_id
        )
        matching_task_id = matching_task["matching_task_id"]
        await client.cd.wait_for_matching(
            session_uuid, mode="trackr", matching_task_id=matching_task_id, timeout=300, interval=3
        )
        print("  [OK] TraCKR matching completed")

        # Step 5: Submit TraCKR Task
        print("  Step 5: Submit TraCKR Task")
        task = await client.cd.run_trackr(
            uploaded_data=uploaded_data,
            transferred_knowledge={"session_uuid": session_uuid, "kg_id": kg_id},
            threshold=0.01,
            matching_task_id=matching_task_id,
        )
        cd_task_id = task["task_id"]
        print(f"  [OK] TraCKR task submitted: {cd_task_id}")

        # Step 6: Poll TraCKR Task Status
        print("  Step 6: Poll TraCKR Task Status")
        await client.cd.wait_for_task(
            cd_task_id, session_uuid=session_uuid, timeout=300, interval=5
        )
        print("  [OK] TraCKR task completed")

        # Step 7: Build DA Input from CD Result
        print("  Step 7: Build DA Input from CD Result")
        task_result = await client.cd.get_task_result(session_uuid, cd_task_id)
        if not task_result.get("result_url"):
            print("  [ERROR] No CD result URL")
            return False
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(task_result["result_url"], timeout=30.0)
            resp.raise_for_status()
            cd_result = resp.json()
        if cd_result.get("status") != "succeeded" or cd_result.get("mode") != "trackr":
            print("  [ERROR] CD result not succeeded or not trackr mode")
            return False
        current_observation, targets = _build_current_observation_and_targets(cd_result)
        if not targets:
            print("  [ERROR] No non-one-hot target found for DA")
            return False
        cd_result_reference = {"session_uuid": session_uuid, "task_id": cd_task_id}
        print(f"  [OK] DA input built (targets: {targets})")

        # Step 8: Submit DA Explain Task
        print("  Step 8: Submit DA Explain Task")
        da_task = await client.da.run_explain(
            session_uuid=session_uuid,
            cd_result_reference=cd_result_reference,
            current_observation=current_observation,
            targets=[{"col": targets[0], "sense": ">", "threshold": 0.8}],
        )
        da_task_id = da_task["task_id"]
        print(f"  [OK] DA explain task submitted: {da_task_id}")

        # Step 9: Poll DA Task Status
        print("  Step 9: Poll DA Task Status")
        await client.da.wait_for_task(session_uuid, da_task_id, timeout=300, interval=5)
        print("  [OK] DA task completed")

        # Step 10: Retrieve and Validate DA Result
        print("  Step 10: Retrieve and Validate DA Result")
        da_result_resp = await client.da.get_task_result(session_uuid, da_task_id)
        if not da_result_resp.get("result_url"):
            print("  [ERROR] No DA result URL")
            return False
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(da_result_resp["result_url"], timeout=30.0)
            resp.raise_for_status()
            da_result = resp.json()
        if da_result.get("status") != "succeeded":
            print("  [ERROR] DA result status not succeeded")
            return False
        if "result" not in da_result:
            print("  [ERROR] DA result missing 'result' key")
            return False
        print("  [OK] DA result validated")

        return True

    except Exception as e:
        print(f"  [ERROR] Workflow failed: {e}")
        return False

    finally:
        # Step 11: Cleanup
        print("  Step 11: Cleanup")
        if session_uuid:
            if da_task_id:
                try:
                    await client.da.delete_task(session_uuid, da_task_id)
                except Exception:
                    pass
            if cd_task_id:
                try:
                    await client.cd.delete_task(session_uuid, cd_task_id)
                except Exception:
                    pass
            if matching_task_id:
                try:
                    await client.cd.delete_trackr_matching(session_uuid, matching_task_id)
                except Exception:
                    pass
            try:
                await client.kg.delete_session(session_uuid)
            except Exception:
                pass
        print("  [OK] Session and trackr matching cleaned up")


async def main() -> None:
    """Run TraCKR + DA example (mirrors test-e2e-trackr-da.sh)."""
    base_url = _resolve_api_base_url()
    if not base_url:
        print("\n[ERROR] Could not determine API Gateway URL. Set CAUSAL_AI_BASE_URL.")
        sys.exit(1)

    api_key, temp_company_name = _get_api_key()
    if not api_key:
        print("\n[ERROR] No API key. Set CAUSAL_AI_API_KEY or use cai-keymgr.")
        sys.exit(1)

    print("=" * 60)
    print("E2E: TraCKR + DA Explain")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print("=" * 60)

    test_data_dir = get_sdk_test_data_dir()
    try:
        async with CausalAIClient(api_key=api_key, base_url=base_url) as client:
            success = await run_trackr_da_workflow(client, test_data_dir)
        print("\n" + "=" * 60)
        if success:
            print("All E2E Tests Passed")
            print("Validated flow: TraCKR CD result -> DA explain")
        else:
            print("Summary: [ERROR] FAILED/SKIPPED")
        print("=" * 60)
    finally:
        if temp_company_name:
            _delete_api_key(temp_company_name)


if __name__ == "__main__":
    asyncio.run(main())
