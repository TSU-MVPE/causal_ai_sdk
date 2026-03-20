# Causal AI SDK Examples

This directory contains example code demonstrating how to use the Causal AI SDK.

## Available Examples

### 1. Knowledge Graph Service Example (`kg_service_example.py`)

Demonstrates the complete Knowledge Graph workflow:

1. Create a session
2. Get presigned upload URL
3. Upload KG data to S3
4. Register KG metadata
5. List all KGs
6. Get specific KG details
7. Download KG data
8. Delete session
9. Verify deletion

### 2. Causal Discovery - MultiCa Example (`cd_multica_example.py`)

Demonstrates the complete MultiCa causal discovery workflow with multiple format support:

1. Create a session
2. Upload multi-dataset data to S3 (CSV/JSON formats)
3. Start column matching computation
4. Wait for matching to complete
5. Test custom matching adjustment (optional)
6. Submit MultiCa task
7. Wait for task to complete
8. Get and validate results
9. Cleanup session

Supports testing with:
- JSON format (single file with multiple datasets)
- CSV format (multiple CSV files or single JSON)
- Multiple CSV files (automatic integration)

### 3. Causal Discovery - TraCKR Example (`cd_trackr_example.py`)

Demonstrates the complete TraCKR causal discovery workflow:

1. Create a session
2. Setup knowledge graph (optional)
3. Upload dataset to S3
4. Submit TraCKR task
5. Wait for task to complete
6. Get and validate results
7. Cleanup session

### 4. End-to-End CD + DA Examples

- `cd_lingam_da_example.py`
- `cd_trackr_da_example.py`
- `cd_multica_da_example.py`

These DA examples use the current DA request format:
`targets=[{"col":"...","sense":">|<|in","threshold":number|[lb,ub]}]`.
Current release enables only one target per request (`len(targets) == 1`).

## Running the Examples

### Prerequisites

Follow the setup steps in the main and component READMEs (install from the **MVP** folder, not from each subdirectory):

- **MVP**: [MVP/README.md](../../README.md) – main project, LocalStack, and development environment
- **Causal AI SDK**: [MVP/causal_ai_sdk/README.md](../README.md) – SDK install (`pip install -e causal_ai_sdk` from MVP)
- **API key manager**: [MVP/api_key_manager/README.md](../../api_key_manager/README.md) – for automatic API key creation (optional; included when installing MVP with `pip install -e .` from MVP)

Then (optional) set environment variables:

```bash
export CAUSAL_AI_API_KEY="your-api-key"   # Optional: auto-created if not set
export CAUSAL_AI_BASE_URL="https://..."  # Optional: auto-detected from Terraform
```

### Automatic Configuration

All examples automatically:

- **Detect LocalStack**: If LocalStack is running (detected via Docker or port 4566), the examples automatically configure:
  - `AWS_ENDPOINT_URL=http://localhost:4566`
  - `AWS_ACCESS_KEY_ID=test`
  - `AWS_SECRET_ACCESS_KEY=test`
  
- **Resolve API Gateway URL**: Automatically resolves from Terraform outputs or environment variables

- **Create Temporary API Keys**: If `CAUSAL_AI_API_KEY` is not set, automatically creates a temporary API key using `cai-keymgr` and cleans it up after execution

### Run the Examples

```bash
cd MVP/causal_ai_sdk/examples

# Knowledge Graph Service
python kg_service_example.py

# MultiCa Causal Discovery
python cd_multica_example.py

# TraCKR Causal Discovery
python cd_trackr_example.py
```

## Notes

- The examples use async/await syntax, so they require Python 3.7+
- The SDK uses async context managers for proper resource management
- All API calls are asynchronous and should be awaited
- Error handling is included to demonstrate proper exception handling
- Temporary API keys are automatically cleaned up after execution
- LocalStack credentials are automatically configured when LocalStack is detected
