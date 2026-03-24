"""Example usage of CD Service - TraCKR mode from Causal AI SDK.

This example demonstrates the complete TraCKR causal discovery workflow with KG integration:
1. Create a session
2. Upload source knowledge graph to KG service
3. Register KG metadata
4. Verify KG exists
5. Upload target dataset to S3 (testing CSV/JSON formats)
6. Start TraCKR column matching computation
7. Wait for matching to complete (polling handled by SDK)
8. Test custom matching adjustment (optional)
9. Submit TraCKR task (with transferred_knowledge)
10. Wait for task to complete (polling handled by SDK)
11. Get and validate results (including knowledge transfer verification)
12. Cleanup session and KG

This example is based on the test-cd-trackr.sh script workflow.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import httpx
from causal_ai_sdk import CausalAIClient
from causal_ai_sdk.models.cd import UploadedData
from helpers import get_api_key_from_env, get_base_url_from_env, get_sdk_test_data_dir

logger = logging.getLogger(__name__)


# Polling is now handled by SDK's wait_for_matching and wait_for_task methods


async def _setup_kg(
    client: CausalAIClient, source_kg_file: Path, format_name: str
) -> tuple[Optional[str], Optional[str]]:
    """Setup knowledge graph for TraCKR workflow.

    Args:
        client (CausalAIClient): Causal AI client instance
        source_kg_file (Path): Path to source KG file
        format_name (str): Format name for display (e.g., "JSON", "CSV")

    Returns:
        tuple[Optional[str], Optional[str]]: (session_uuid, kg_id) or (None, None) if failed
    """
    try:
        # Step 1: Create session
        print(f"  [{format_name}] Step 1: Create Session")
        session = await client.kg.init_session()
        session_uuid = session["uuid"]
        print(f"  [{format_name}] [OK] Session created: {session_uuid}")

        # Step 2-3: Upload source knowledge graph
        print(f"  [{format_name}] Step 2-3: Upload Source Knowledge Graph")
        if not source_kg_file.exists():
            print(f"  [{format_name}]   Source KG file not found, creating sample data...")
            sample_kg_data = {
                "title": "Source Domain Knowledge Graph",
                "columns": ["X1", "X2", "X3", "X4"],
                "data": [
                    [0.0, 0.85, 0.0, 0.0],
                    [0.0, 0.0, 0.75, 0.0],
                    [0.0, 0.0, 0.0, 0.65],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            }
            source_kg_file.parent.mkdir(parents=True, exist_ok=True)
            with open(source_kg_file, "w") as f:
                json.dump(sample_kg_data, f)

        kg = await client.kg.upload_kg_from_file(
            session_uuid=session_uuid, file_path=source_kg_file
        )
        print(f"  [{format_name}] [OK] KG uploaded and registered: {kg['id']}")
        print(f"  [{format_name}]   Title: {kg['title']}")
        kg_id = kg["id"]

        # Step 4: Verify KG exists
        print(f"  [{format_name}] Step 4: Verify Knowledge Graph")
        kg_detail = await client.kg.get_kg(session_uuid, kg_id)
        print(f"  [{format_name}] [OK] KG verified in storage")
        cols = kg_detail.get("columns", [])
        print(f"  [{format_name}]   Columns: {cols}")
        row_count = kg_detail.get("row_count", "N/A")
        print(f"  [{format_name}]   Row count: {row_count}")

        return session_uuid, kg_id

    except Exception as e:
        print(f"  [{format_name}] [ERROR] KG setup failed: {e}")
        return None, None


async def _run_trackr_workflow(
    client: CausalAIClient,
    session_uuid: str,
    uploaded_data: UploadedData,
    kg_id: str,
    format_name: str,
) -> tuple[bool, Optional[str]]:
    """Run the complete TraCKR workflow for uploaded data.

    Args:
        client (CausalAIClient): Causal AI client instance
        session_uuid (str): Session UUID
        uploaded_data (UploadedData): From upload_data_for_trackr
        kg_id (str): Source knowledge graph ID
        format_name (str): Format name for display (e.g., "JSON", "CSV")

    Returns:
        tuple[bool, Optional[str]]: (success, matching_task_id for cleanup or None)
    """
    matching_task_id: Optional[str] = None
    try:
        # Step 6: Start TraCKR column matching
        print(f"  [{format_name}] Step 6: Start TraCKR Column Matching")
        matching_task = await client.cd.start_trackr_matching(
            uploaded_data=uploaded_data, source_kg_id=kg_id
        )
        matching_task_id = matching_task["matching_task_id"]
        print(f"  [{format_name}] [OK] Matching computation started: {matching_task_id}")

        # Step 7: Wait for matching to complete
        print(f"  [{format_name}] Step 7: Wait for Matching to Complete")
        matching_result = await client.cd.wait_for_matching(
            session_uuid,
            mode="trackr",
            timeout=300,
            interval=3,
            matching_task_id=matching_task_id,
        )
        print(f"  [{format_name}] [OK] Matching completed")
        knowledge_coverage = matching_result.get("knowledge_coverage")
        if knowledge_coverage:
            print(f"  [{format_name}]   Knowledge coverage: {knowledge_coverage}")
        current = matching_result.get("current_matched") or {}
        num_matched = len([v for v in current.values() if v])
        print(f"  [{format_name}]   Matched columns: {num_matched}")

        # Step 8: Test custom matching adjustment (optional)
        print(f"  [{format_name}] Step 8: Test Custom Matching Adjustment")
        current_matching = matching_result.get("current_matched") or {}
        if current_matching:
            custom_matching = current_matching.copy()
            first_col = list(custom_matching.keys())[0]
            original_match_value = custom_matching[first_col]
            custom_matching[first_col] = ""  # Remove match
            print(
                f"  [{format_name}]   Removing match for {first_col}: "
                f"{original_match_value} -> (no match)"
            )

            await client.cd.set_trackr_matching(
                session_uuid, custom_matching, matching_task_id=matching_task_id
            )
            print(f"  [{format_name}] [OK] Custom matching applied")

            # Verify the change
            updated_result = await client.cd.get_trackr_matching(session_uuid, matching_task_id)
            if (updated_result.get("current_matched") or {}).get(first_col) == "":
                print(f"  [{format_name}] [OK] Custom matching verified")
            else:
                print(
                    f"  [{format_name}] [WARN] Custom matching may not have been applied correctly"
                )
        else:
            print(f"  [{format_name}]   No matching available to adjust, skipping...")

        # Step 9: Submit TraCKR task
        print(f"  [{format_name}] Step 9: Submit TraCKR Task")
        transferred_knowledge = {
            "session_uuid": session_uuid,
            "kg_id": kg_id,
        }
        task = await client.cd.run_trackr(
            uploaded_data=uploaded_data,
            transferred_knowledge=transferred_knowledge,
            threshold=0.01,
            matching_task_id=matching_task_id,
        )
        run_task_id = task["task_id"]
        print(f"  [{format_name}] [OK] Task submitted: {run_task_id}")
        print(f"  [{format_name}]   Using KG: {kg_id}")

        # Step 10: Wait for task to complete (use run task_id, not upload task_id)
        print(f"  [{format_name}] Step 10: Wait for Task to Complete")
        await client.cd.wait_for_task(
            run_task_id, session_uuid=session_uuid, timeout=300, interval=5
        )
        print(f"  [{format_name}] [OK] Task completed successfully")

        # Step 11: Get and validate results
        print(f"  [{format_name}] Step 11: Retrieve Results")
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

        # Check TraCKR-specific metadata
        if "kg_id" in result_data and "matching_used" in result_data:
            print(f"  [{format_name}] [OK] Results retrieved and validated")

            # Print detailed results
            print(f"  [{format_name}]")
            print(f"  [{format_name}] ========================================")
            print(f"  [{format_name}] Results")
            print(f"  [{format_name}] ========================================")
            print(f"  [{format_name}] Mode: {result_data.get('mode', 'N/A')}")
            print(f"  [{format_name}] Variables: {', '.join(result_data.get('feature_names', []))}")
            print(f"  [{format_name}] Number of datasets: {result_data.get('num_datasets', 'N/A')}")
            print(
                f"  [{format_name}] Number of variables: {result_data.get('num_variables', 'N/A')}"
            )
            print(f"  [{format_name}]")

            # Print adjacency matrix
            adjacency_matrices = result_data.get("adjacency_matrices", [])
            if adjacency_matrices:
                print(f"  [{format_name}] Adjacency Matrix:")
                for row in adjacency_matrices:
                    print(f"  [{format_name}]   {row}")
                print(f"  [{format_name}]")

            print(f"  [{format_name}] [OK] Result structure is valid")
            print(f"  [{format_name}]")

            # Print knowledge transfer verification
            print(f"  [{format_name}] === Knowledge Transfer Verification ===")
            print(f"  [{format_name}]   KG ID: {result_data['kg_id']}")
            print(f"  [{format_name}]   Matching state: {result_data['matching_used']}")
            print(f"  [{format_name}]   [OK] Verified: TraCKR executed with knowledge transfer")
            print(f"  [{format_name}]   - Used knowledge graph: {result_data['kg_id']}")
            print(
                f"  [{format_name}]   - Applied column matching from: "
                f"{result_data['matching_used']}"
            )
            print(f"  [{format_name}] [SUCCESS] Results validated with knowledge transfer metadata")
        else:
            missing_fields = []
            if "kg_id" not in result_data:
                missing_fields.append("kg_id")
            if "matching_used" not in result_data:
                missing_fields.append("matching_used")
            print(
                f"  [{format_name}] [ERROR] Knowledge transfer metadata missing: "
                f"{', '.join(missing_fields)}"
            )
            return False, matching_task_id

        return True, matching_task_id

    except Exception as e:
        print(f"  [{format_name}] [ERROR] Workflow failed: {e}")
        return False, matching_task_id


async def test_trackr_json(
    client: CausalAIClient, test_data_dir: Path
) -> tuple[bool, Optional[str]]:
    """Test TraCKR workflow with JSON format.

    Args:
        client (CausalAIClient): Causal AI client instance
        test_data_dir (Path): Directory containing test data files

    Returns:
        tuple[bool, Optional[str]]: (success, session_uuid) or (False, None) if file not found
    """
    source_kg_file = test_data_dir / "kg" / "source_graph.json"
    trackr_json = test_data_dir / "cd" / "trackr_target_data.json"

    if not trackr_json.exists():
        print("  [JSON] Test file not found, skipping...")
        return False, None

    print("\n" + "=" * 60)
    print("Testing TraCKR with JSON Format")
    print("=" * 60)

    try:
        # Setup KG
        session_uuid, kg_id = await _setup_kg(client, source_kg_file, "JSON")
        if not session_uuid or not kg_id:
            return False, None

        # Step 5: Upload target data
        print("  [JSON] Step 5: Upload Target Data")
        uploaded_data = await client.cd.upload_data_for_trackr(session_uuid, trackr_json)
        print("  [JSON] [OK] JSON file uploaded successfully")
        print(f"  [JSON]   Task ID: {uploaded_data.task_id}")

        # Run workflow
        success, matching_task_id = await _run_trackr_workflow(
            client, session_uuid, uploaded_data, kg_id, "JSON"
        )

        # Cleanup (follow test-cd-trackr.sh: delete matching then session)
        print("  [JSON] Step 12: Cleanup")
        if matching_task_id:
            await client.cd.delete_trackr_matching(session_uuid, matching_task_id)
        print("  [JSON] [OK] Matching state deleted")
        await client.kg.delete_session(session_uuid)
        print("  [JSON] [OK] Session deleted")

        return success, session_uuid

    except Exception as e:
        print(f"  [JSON] [ERROR] Test failed: {e}")
        return False, None


async def test_trackr_csv(
    client: CausalAIClient, test_data_dir: Path
) -> tuple[bool, Optional[str]]:
    """Test TraCKR workflow with CSV format.

    Args:
        client (CausalAIClient): Causal AI client instance
        test_data_dir (Path): Directory containing test data files

    Returns:
        tuple[bool, Optional[str]]: (success, session_uuid) or (False, None) if file not found
    """
    source_kg_file = test_data_dir / "kg" / "source_graph.json"
    trackr_csv = test_data_dir / "cd" / "trackr_target_data.csv"

    if not trackr_csv.exists():
        print("  [CSV] Test file not found, skipping...")
        return False, None

    print("\n" + "=" * 60)
    print("Testing TraCKR with CSV Format")
    print("=" * 60)

    try:
        # Setup KG
        session_uuid, kg_id = await _setup_kg(client, source_kg_file, "CSV")
        if not session_uuid or not kg_id:
            return False, None

        # Step 5: Upload target data
        print("  [CSV] Step 5: Upload Target Data")
        uploaded_data = await client.cd.upload_data_for_trackr(session_uuid, trackr_csv)
        print("  [CSV] [OK] CSV file uploaded successfully")
        print(f"  [CSV]   Task ID: {uploaded_data.task_id}")

        # Run workflow
        success, matching_task_id = await _run_trackr_workflow(
            client, session_uuid, uploaded_data, kg_id, "CSV"
        )

        # Cleanup (follow test-cd-trackr.sh: delete matching then session)
        print("  [CSV] Step 12: Cleanup")
        if matching_task_id:
            await client.cd.delete_trackr_matching(session_uuid, matching_task_id)
        print("  [CSV] [OK] Matching state deleted")
        await client.kg.delete_session(session_uuid)
        print("  [CSV] [OK] Session deleted")

        return success, session_uuid

    except Exception as e:
        print(f"  [CSV] [ERROR] Test failed: {e}")
        return False, None


async def main():
    """Main example function demonstrating TraCKR workflow with multiple format tests."""
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
        print("TraCKR Causal Discovery Service Example")
        print("=" * 60)
        print(f"Base URL: {base_url}")
        print(f"API Key: {api_key[:15]}..." if len(api_key) > 15 else f"API Key: {api_key}")
        print("=" * 60)

    # Get test data directory
    test_data_dir = get_sdk_test_data_dir()

    # Use async context manager for proper resource management
    async with CausalAIClient(api_key=api_key, base_url=base_url) as client:
        results = {}

        # Test JSON format
        success, session_uuid = await test_trackr_json(client, test_data_dir)
        results["JSON"] = success

        # Test CSV format
        success, session_uuid = await test_trackr_csv(client, test_data_dir)
        results["CSV"] = success

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
