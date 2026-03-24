"""Example usage of CD Service - MultiCa mode from Causal AI SDK.

This example demonstrates the complete MultiCa causal discovery workflow:
1. Create a session
2. Upload multi-dataset data to S3 (testing CSV/JSON formats)
3. Start column matching computation
4. Wait for matching to complete (polling handled by SDK)
5. Test custom matching adjustment (optional)
6. Submit MultiCa task
7. Wait for task to complete (polling handled by SDK)
8. Get and validate results
9. Cleanup session

This example is based on the test-cd-multica.sh script workflow.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Union

import httpx
from causal_ai_sdk import CausalAIClient
from causal_ai_sdk.models.cd import UploadedData
from helpers import get_api_key_from_env, get_base_url_from_env, get_sdk_test_data_dir

logger = logging.getLogger(__name__)


async def _run_multica_workflow(
    client: CausalAIClient,
    session_uuid: str,
    uploaded_data: UploadedData,
    format_name: str,
) -> tuple[bool, Optional[str]]:
    """Run the complete MultiCa workflow for uploaded data.

    Args:
        client (CausalAIClient): Causal AI client instance
        session_uuid (str): Session UUID
        uploaded_data (UploadedData): From upload_data_for_multica
        format_name (str): Format name for display (e.g., "JSON", "CSV")

    Returns:
        tuple[bool, Optional[str]]: (success, matching_task_id for cleanup or None)
    """
    matching_task_id: Optional[str] = None
    try:
        # Step 3: Start column matching
        print(f"\n  [{format_name}] Step 3: Start Column Matching")
        matching_task = await client.cd.start_multica_matching(uploaded_data)
        matching_task_id = matching_task["matching_task_id"]
        print(f"  [{format_name}] [OK] Matching computation started: {matching_task_id}")

        # Step 4: Wait for matching to complete
        print(f"  [{format_name}] Step 4: Wait for Matching to Complete")
        matching_result = await client.cd.wait_for_matching(
            session_uuid, mode="multica", matching_task_id=matching_task_id, timeout=300, interval=3
        )
        print(f"  [{format_name}] [OK] Matching completed")
        if matching_result.get("match_rate"):
            print(f"  [{format_name}]   Match rate: {matching_result.get('match_rate')}")

        # Step 5: Test custom matching adjustment (optional)
        print(f"  [{format_name}] Step 5: Test Custom Matching Adjustment")
        current_matching = matching_result.get("current_matched") or {}
        if current_matching:
            custom_matching = current_matching.copy()
            first_col = list(custom_matching.keys())[0]
            custom_matching[first_col] = first_col  # Self-match
            await client.cd.set_multica_matching(
                session_uuid, custom_matching, matching_task_id=matching_task_id
            )
            print(f"  [{format_name}] [OK] Custom matching applied")

        # Step 6: Submit MultiCa task (matching_task_id required by API)
        print(f"  [{format_name}] Step 6: Submit MultiCa Task")
        task = await client.cd.run_multica(
            uploaded_data, threshold=0.01, matching_task_id=matching_task_id
        )
        run_task_id = task["task_id"]
        print(f"  [{format_name}] [OK] Task submitted: {run_task_id}")

        # Step 7: Wait for task to complete (use run task_id, not upload task_id)
        print(f"  [{format_name}] Step 7: Wait for Task to Complete")
        await client.cd.wait_for_task(
            run_task_id, session_uuid=session_uuid, timeout=300, interval=5
        )
        print(f"  [{format_name}] [OK] Task completed successfully")

        # Step 8: Get and validate results
        print(f"  [{format_name}] Step 8: Retrieve Results")
        task_result = await client.cd.get_task_result(session_uuid, run_task_id)

        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(task_result["result_url"], timeout=30.0)
            response.raise_for_status()
            result_data = response.json()

        # Validate result structure
        required_fields = ["mode", "feature_names", "adjacency_matrices"]
        if not all(field in result_data for field in required_fields):
            missing = [f for f in required_fields if f not in result_data]
            print(f"  [{format_name}] [ERROR] Result structure incomplete (missing: {missing})")
            return False, matching_task_id

        print(f"  [{format_name}] [OK] Results retrieved and validated")
        print(f"  [{format_name}]   Mode: {result_data.get('mode', 'N/A')}")
        print(f"  [{format_name}]   Variables: {len(result_data.get('feature_names', []))}")
        print(f"  [{format_name}]   Datasets: {result_data.get('num_datasets', 'N/A')}")

        return True, matching_task_id

    except Exception as e:
        print(f"  [{format_name}] [ERROR] Workflow failed: {e}")
        return False, matching_task_id


async def test_multica_json(
    client: CausalAIClient, test_data_dir: Path
) -> tuple[bool, Optional[str]]:
    """Test MultiCa workflow with JSON format.

    Note: JSON file can be used alone if it contains multiple datasets in the "data" array.

    Args:
        client (CausalAIClient): Causal AI client instance
        test_data_dir (Path): Directory containing test data files

    Returns:
        tuple[bool, Optional[str]]: (success, session_uuid) or (False, None) if file not found
    """
    json_file = test_data_dir / "multica_datasets.json"
    if not json_file.exists():
        print("  [JSON] Test file not found, skipping...")
        return False, None

    print("\n" + "=" * 60)
    print("Testing MultiCa with JSON Format")
    print("=" * 60)
    print("  [JSON] Note: JSON file can contain multiple datasets in 'data' array")

    try:
        # Create session
        print("  [JSON] Step 1: Create Session")
        session = await client.kg.init_session()
        session_uuid = session["uuid"]
        print(f"  [JSON] [OK] Session created: {session_uuid}")

        # Upload data
        print("  [JSON] Step 2: Upload Data")
        print("  [JSON] Note: JSON file with multiple datasets can be used alone")
        uploaded_data = await client.cd.upload_data_for_multica(session_uuid, json_file)
        print("  [JSON] [OK] JSON file uploaded successfully")
        print(f"  [JSON]   Task ID: {uploaded_data.task_id}")

        # Run workflow
        success, matching_task_id = await _run_multica_workflow(
            client, session_uuid, uploaded_data, "JSON"
        )

        # Cleanup (follow test-cd-multica.sh: delete matching then session)
        print("  [JSON] Step 9: Cleanup")
        if matching_task_id:
            await client.cd.delete_multica_matching(session_uuid, matching_task_id)
        print("  [JSON] [OK] Matching state deleted")
        await client.kg.delete_session(session_uuid)
        print("  [JSON] [OK] Session deleted")

        return success, session_uuid

    except Exception as e:
        print(f"  [JSON] [ERROR] Test failed: {e}")
        # Check if it's a validation error about file count
        if "at least 2 datasets" in str(e).lower() or "at least 2" in str(e).lower():
            print("  [JSON] Note: JSON file must contain at least 2 datasets in the 'data' array.")
        return False, None


async def test_multica_csv(
    client: CausalAIClient, test_data_dir: Path
) -> tuple[bool, Optional[str]]:
    """Test MultiCa workflow with CSV format (using multiple CSV files).

    Args:
        client (CausalAIClient): Causal AI client instance
        test_data_dir (Path): Directory containing test data files

    Returns:
        tuple[bool, Optional[str]]: (success, session_uuid) or (False, None) if files not found
    """
    # Try to use multiple CSV files first
    dataset1_csv = test_data_dir / "multica_dataset1.csv"
    dataset2_csv = test_data_dir / "multica_dataset2.csv"

    # If multiple CSV files don't exist, try to create them from JSON
    json_file = test_data_dir / "multica_datasets.json"
    if json_file.exists() and (not dataset1_csv.exists() or not dataset2_csv.exists()):
        try:
            import csv

            with open(json_file, "r") as f:
                multica_data = json.load(f)

            for i, dataset in enumerate(multica_data.get("data", []), 1):
                csv_file = test_data_dir / f"multica_dataset{i}.csv"
                if not csv_file.exists():
                    with open(csv_file, "w", newline="") as cf:
                        writer = csv.writer(cf)
                        writer.writerow(dataset["columns"])
                        for row in dataset["data"]:
                            writer.writerow(row)
            print("  [CSV] Created separate CSV files for testing")
        except Exception as e:
            print(f"  [CSV] Could not create CSV files: {e}")

    # Use multiple CSV files if available, otherwise fall back to JSON
    csv_files: list[Union[str, Path]]
    if dataset1_csv.exists() and dataset2_csv.exists():
        csv_files = [dataset1_csv, dataset2_csv]
        use_multiple = True
    elif json_file.exists():
        # Fall back to JSON if CSV files not available
        csv_files = [json_file]
        use_multiple = False
    else:
        print("  [CSV] Test files not found, skipping...")
        return False, None

    print("\n" + "=" * 60)
    print("Testing MultiCa with CSV Format")
    print("=" * 60)

    try:
        # Create session
        print("  [CSV] Step 1: Create Session")
        session = await client.kg.init_session()
        session_uuid = session["uuid"]
        print(f"  [CSV] [OK] Session created: {session_uuid}")

        # Upload data
        print("  [CSV] Step 2: Upload Data")
        if use_multiple:
            uploaded_data = await client.cd.upload_data_for_multica(session_uuid, csv_files)
            print("  [CSV] [OK] Multiple CSV files uploaded successfully")
        else:
            # JSON file contains multiple datasets, so it should work
            uploaded_data = await client.cd.upload_data_for_multica(session_uuid, csv_files[0])
            print("  [CSV] [OK] CSV/JSON file uploaded successfully")
        print(f"  [CSV]   Task ID: {uploaded_data.task_id}")

        # Run workflow
        success, matching_task_id = await _run_multica_workflow(
            client, session_uuid, uploaded_data, "CSV"
        )

        # Cleanup (follow test-cd-multica.sh: delete matching then session)
        print("  [CSV] Step 9: Cleanup")
        if matching_task_id:
            await client.cd.delete_multica_matching(session_uuid, matching_task_id)
        print("  [CSV] [OK] Matching state deleted")
        await client.kg.delete_session(session_uuid)
        print("  [CSV] [OK] Session deleted")

        return success, session_uuid

    except Exception as e:
        print(f"  [CSV] [ERROR] Test failed: {e}")
        return False, None


async def test_multica_multiple_csv(
    client: CausalAIClient, test_data_dir: Path
) -> tuple[bool, Optional[str]]:
    """Test MultiCa workflow with multiple CSV files (automatic integration).

    Args:
        client (CausalAIClient): Causal AI client instance
        test_data_dir (Path): Directory containing test data files

    Returns:
        tuple[bool, Optional[str]]: (success, session_uuid) or (False, None) if files not found
    """
    dataset1_csv = test_data_dir / "multica_dataset1.csv"
    dataset2_csv = test_data_dir / "multica_dataset2.csv"

    # Create CSV files from JSON if they don't exist
    json_file = test_data_dir / "multica_datasets.json"
    if json_file.exists() and (not dataset1_csv.exists() or not dataset2_csv.exists()):
        try:
            import csv

            with open(json_file, "r") as f:
                multica_data = json.load(f)

            for i, dataset in enumerate(multica_data.get("data", []), 1):
                csv_file = test_data_dir / f"multica_dataset{i}.csv"
                if not csv_file.exists():
                    with open(csv_file, "w", newline="") as cf:
                        writer = csv.writer(cf)
                        writer.writerow(dataset["columns"])
                        for row in dataset["data"]:
                            writer.writerow(row)
            print("  [Multiple CSV] Created separate CSV files for testing")
        except Exception as e:
            print(f"  [Multiple CSV] Could not create CSV files: {e}")

    if not dataset1_csv.exists() or not dataset2_csv.exists():
        print("  [Multiple CSV] Test files not found, skipping...")
        return False, None

    print("\n" + "=" * 60)
    print("Testing MultiCa with Multiple CSV Files")
    print("=" * 60)

    try:
        # Create session
        print("  [Multiple CSV] Step 1: Create Session")
        session = await client.kg.init_session()
        session_uuid = session["uuid"]
        print(f"  [Multiple CSV] [OK] Session created: {session_uuid}")

        # Upload data
        print("  [Multiple CSV] Step 2: Upload Data")
        multiple_files: list[Union[str, Path]] = [dataset1_csv, dataset2_csv]
        uploaded_data = await client.cd.upload_data_for_multica(session_uuid, multiple_files)
        print("  [Multiple CSV] [OK] Multiple CSV files uploaded successfully")
        print(f"  [Multiple CSV]   Task ID: {uploaded_data.task_id}")
        print("  [Multiple CSV]   Files automatically integrated into MultiCa format")

        # Run workflow
        success, matching_task_id = await _run_multica_workflow(
            client, session_uuid, uploaded_data, "Multiple CSV"
        )

        # Cleanup (follow test-cd-multica.sh: delete matching then session)
        print("  [Multiple CSV] Step 9: Cleanup")
        if matching_task_id:
            await client.cd.delete_multica_matching(session_uuid, matching_task_id)
        print("  [Multiple CSV] [OK] Matching state deleted")
        await client.kg.delete_session(session_uuid)
        print("  [Multiple CSV] [OK] Session deleted")

        return success, session_uuid

    except Exception as e:
        print(f"  [Multiple CSV] [ERROR] Test failed: {e}")
        return False, None


async def main():
    """Main example function demonstrating MultiCa workflow with multiple format tests."""
    # Get API Gateway URL from Terraform or environment
    base_url = get_base_url_from_env()
    if not base_url:
        print("[ERROR] No API base URL. Please set CAUSAL_AI_BASE_URL environment variable.")
        sys.exit(1)

    api_key = get_api_key_from_env()
    if not api_key:
        print(
            "\n[ERROR] No API key available.\n" "Please set CAUSAL_AI_API_KEY environment variable."
        )
        sys.exit(1)

        print("=" * 60)
        print("MultiCa Causal Discovery Service Example")
        print("=" * 60)
        print(f"Base URL: {base_url}")
        print(f"API Key: {api_key[:15]}..." if len(api_key) > 15 else f"API Key: {api_key}")
        print("=" * 60)

    # Get test data directory
    test_data_dir = get_sdk_test_data_dir() / "cd"

    # Use async context manager for proper resource management
    async with CausalAIClient(api_key=api_key, base_url=base_url) as client:
        results = {}

        # Test JSON format
        success, session_uuid = await test_multica_json(client, test_data_dir)
        results["JSON"] = success

        # Test CSV format
        success, session_uuid = await test_multica_csv(client, test_data_dir)
        results["CSV"] = success

        # Test multiple CSV files
        success, session_uuid = await test_multica_multiple_csv(client, test_data_dir)
        results["Multiple CSV"] = success

        # Print summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        for format_name, success in results.items():
            status = "[OK] PASSED" if success else "[ERROR] FAILED/SKIPPED"
            print(f"  {format_name}: {status}")

        total_tests = len(results)
        passed_tests = sum(1 for s in results.values() if s)
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
