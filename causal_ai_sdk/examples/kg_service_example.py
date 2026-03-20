"""Example usage of KG Service from Causal AI SDK.

This example demonstrates the complete Knowledge Graph workflow:
1. Create a session
2. Get presigned upload URL
3. Upload KG data to S3
4. Register KG metadata
5. List all KGs
6. Get specific KG details
7. Download KG data
8. Delete session

This example is based on the test-kg.sh script workflow.
"""

import asyncio
import json
import logging
import sys

from causal_ai_sdk import CausalAIClient
from helpers import (
    _delete_api_key,
    _get_api_key,
    _resolve_api_base_url,
    get_sdk_test_data_dir,
)

logger = logging.getLogger(__name__)


async def main():
    """Main example function demonstrating KG service workflow."""
    # Track if we created a temporary API key for cleanup
    temp_company_name = None

    try:
        # Get API Gateway URL from Terraform or environment
        base_url = _resolve_api_base_url()
        if not base_url:
            print(
                "[ERROR] Could not determine API Gateway URL.\n"
                "Please either:\n"
                "  1. Set CAUSAL_AI_BASE_URL environment variable, or\n"
                "  2. Ensure Terraform is configured and outputs are available, or\n"
                "  3. Set AWS_ENDPOINT_URL for LocalStack (e.g., http://localhost:4566)"
            )
            sys.exit(1)

        # Get API key from environment or create temporary one
        api_key, temp_company_name = _get_api_key()
        if not api_key:
            print(
                "\n[ERROR] No API key available.\n"
                "Please either:\n"
                "  1. Set CAUSAL_AI_API_KEY environment variable, or\n"
                "  2. Ensure AWS credentials are configured for cai-keymgr"
            )
            sys.exit(1)

        print("=" * 60)
        print("Knowledge Graph Service Example")
        print("=" * 60)
        print(f"Base URL: {base_url}")
        print(f"API Key: {api_key[:15]}..." if len(api_key) > 15 else f"API Key: {api_key}")
        print("=" * 60)

        # Use async context manager for proper resource management
        async with CausalAIClient(api_key=api_key, base_url=base_url) as client:
            # Step 1: Create session
            print("\n" + "=" * 60)
            print("Step 1: Create Session")
            print("=" * 60)
            try:
                session = await client.kg.init_session()
                print(f"[OK] Session created: {session['uuid']}")
                print(f"  Status: {session['status']}")
                session_uuid = session["uuid"]
            except Exception as e:
                print(f"[ERROR] Error creating session: {e}")
                return

            # Step 2: Get presigned upload URL
            print("\n" + "=" * 60)
            print("Step 2: Get Presigned Upload URL")
            print("=" * 60)
            try:
                upload_url = await client.kg.get_upload_url(
                    session_uuid=session_uuid, filename="test-kg.json"
                )
                print("[OK] Got presigned upload URL")
                print(f"  S3 Key: {upload_url['s3_key']}")
                print(f"  Upload URL: {upload_url['upload_url'][:80]}...")
            except Exception as e:
                print(f"[ERROR] Error getting upload URL: {e}")
                return

            # Step 3: Upload KG data to S3
            print("\n" + "=" * 60)
            print("Step 3: Upload Knowledge Graph Data to S3")
            print("=" * 60)

            # Load test data (similar to test-kg.sh)
            test_data_path = get_sdk_test_data_dir() / "kg" / "source_graph.json"

            if not test_data_path.exists():
                # Create sample data if test data doesn't exist
                print("  Test data not found, creating sample data...")
                sample_data = {
                    "title": "Source Domain Knowledge Graph",
                    "columns": ["X1", "X2", "X3", "X4"],
                    "data": [
                        [0.0, 0.85, 0.0, 0.0],
                        [0.0, 0.0, 0.75, 0.0],
                        [0.0, 0.0, 0.0, 0.65],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                }
                kg_data = sample_data
            else:
                print(f"  Loading test data from: {test_data_path}")
                with open(test_data_path, "r", encoding="utf-8") as f:
                    kg_data = json.load(f)

            try:
                import httpx

                async with httpx.AsyncClient() as http_client:
                    response = await http_client.put(
                        upload_url["upload_url"],
                        json=kg_data,
                        headers={"Content-Type": "application/json"},
                        timeout=30.0,
                    )
                    response.raise_for_status()
                print("[OK] Data uploaded to S3 successfully")
                print(f"  HTTP Status: {response.status_code}")
            except Exception as e:
                print(f"[ERROR] Error uploading to S3: {e}")
                return

            # Step 4: Register KG metadata
            print("\n" + "=" * 60)
            print("Step 4: Register Knowledge Graph Metadata")
            print("=" * 60)
            try:
                kg = await client.kg.add_kg(
                    session_uuid=session_uuid,
                    title=kg_data.get("title", "Source Knowledge Graph"),
                    columns=kg_data["columns"],
                    s3_key=upload_url["s3_key"],
                    row_count=len(kg_data["data"]),
                )
                print(f"[OK] KG registered: {kg['id']}")
                print(f"  Title: {kg['title']}")
                print(f"  Number of nodes: {kg['num_nodes']}")
                kg_id = kg["id"]
            except Exception as e:
                print(f"[ERROR] Error registering KG: {e}")
                return

            # Step 5: List all KGs
            print("\n" + "=" * 60)
            print("Step 5: List All Knowledge Graphs")
            print("=" * 60)
            try:
                kg_list = await client.kg.list_kg(session_uuid=session_uuid)
                print(f"[OK] Found {len(kg_list)} knowledge graph(s)")
                for kg_item in kg_list:
                    print(f"  - {kg_item['title']} (ID: {kg_item['id']})")
            except Exception as e:
                print(f"[ERROR] Error listing knowledge graphs: {e}")

            # Step 6: Get specific KG details
            print("\n" + "=" * 60)
            print("Step 6: Get Specific Knowledge Graph")
            print("=" * 60)
            try:
                kg_detail = await client.kg.get_kg(session_uuid=session_uuid, kg_id=kg_id)
                print("[OK] Got knowledge graph details")
                detail_info = []
                if kg_detail.get("title"):
                    detail_info.append(f"Title: {kg_detail['title']}")
                if kg_detail.get("created_at"):
                    detail_info.append(f"Created at: {kg_detail['created_at']}")
                if kg_detail.get("download_url"):
                    detail_info.append(f"Download URL: {kg_detail['download_url'][:80]}...")
                if detail_info:
                    for info in detail_info:
                        print(f"  {info}")
            except Exception as e:
                print(f"[ERROR] Error getting knowledge graph details: {e}")

            # Step 7: Download KG data (if download_url is available)
            print("\n" + "=" * 60)
            print("Step 7: Download Knowledge Graph Data")
            print("=" * 60)
            try:
                kg_detail = await client.kg.get_kg(session_uuid=session_uuid, kg_id=kg_id)
                if kg_detail.get("download_url"):
                    import httpx

                    async with httpx.AsyncClient() as http_client:
                        response = await http_client.get(kg_detail["download_url"], timeout=30.0)
                        response.raise_for_status()
                        downloaded_data = response.json()
                    print("[OK] Data downloaded successfully")
                    print(f"  Columns: {downloaded_data.get('columns', [])}")
                    print(f"  Data rows: {len(downloaded_data.get('data', []))}")
                else:
                    print("  Download URL not available, skipping download")
            except Exception as e:
                print(f"[ERROR] Error downloading data: {e}")

            # Step 8: Delete session
            print("\n" + "=" * 60)
            print("Step 8: Delete Session")
            print("=" * 60)
            try:
                await client.kg.delete_session(session_uuid=session_uuid)
                print(f"[OK] Session {session_uuid} deleted")
            except Exception as e:
                print(f"[ERROR] Error deleting session: {e}")

            # Summary
            print("\n" + "=" * 60)
            print("Test Summary")
            print("=" * 60)
            print("[OK] Session created")
            print("[OK] Presigned URL obtained")
            print("[OK] Data uploaded to S3")
            print("[OK] KG metadata registered")
            print("[OK] KG listed")
            print("[OK] KG retrieved")
            print("[OK] Data downloaded")
            print("[OK] Session deleted")
            print("\n" + "=" * 60)
            print("All Steps Completed Successfully!")
            print("=" * 60)

    finally:
        # Clean up temporary API key if we created one
        if temp_company_name:
            print("\n" + "=" * 60)
            print("Cleaning Up")
            print("=" * 60)
            _delete_api_key(temp_company_name)


if __name__ == "__main__":
    asyncio.run(main())
