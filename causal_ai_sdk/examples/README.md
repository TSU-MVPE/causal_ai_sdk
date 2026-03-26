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

### 2. Causal Discovery - MultiCa Example (`cd_multica_example.py`)

Demonstrates the complete MultiCa causal discovery workflow with multiple format support:

1. Create a session
2. Upload multi-dataset data to S3 (CSV/JSON formats)
3. Start column matching computation
4. Wait for matching to complete
5. Test custom matching adjustment
6. Submit MultiCa task
7. Wait for task to complete
8. Get and validate results
9. Delete MultiCa matching state, then delete the session

Supports testing with:
- JSON format (single file with multiple datasets)
- CSV format (multiple CSV files or single JSON)
- Multiple CSV files (automatic integration)

### 3. Causal Discovery - TraCKR Example (`cd_trackr_example.py`)

Demonstrates the complete TraCKR causal discovery workflow:

1. Create a session
2. Upload source knowledge graph to the KG service
3. Register KG metadata
4. Verify the KG exists
5. Upload target dataset to S3
6. Start TraCKR column matching
7. Wait for matching to complete
8. Test custom matching adjustment
9. Submit TraCKR task
10. Wait for the CD task to complete
11. Get and validate results
12. Delete TraCKR matching state, then delete the session

### 4. End-to-End CD + DA Examples

- `cd_multica_da_example.py`
- `cd_trackr_da_example.py`
- `cd_lingam_da_example.py`

## Running the Examples

### Prerequisites

Install the SDK and required optional dependencies first:

- **Python 3.10+** (required by the SDK package; see `pyproject.toml`)
- **Causal AI SDK**: [SDK README](../README.md)
- **python-dotenv** (for `.env` loading in examples): `pip install python-dotenv`
- **API key**: required via environment variable (`CAUSAL_AI_API_KEY`)

Set environment variables (or place the same values in `examples/.env`):

```bash
export CAUSAL_AI_API_KEY="your-api-key"   # Required
export CAUSAL_AI_BASE_URL="https://..."   # Required
```

### Configuration Behavior

All examples:

- **Use provided API base URL**: Read `CAUSAL_AI_BASE_URL` from environment or `examples/.env`
- **Use provided API key**: Read `CAUSAL_AI_API_KEY` from environment or `examples/.env`

### Run the Examples

```bash
cd examples

# Knowledge Graph Service
python kg_service_example.py

# MultiCa Causal Discovery
python cd_multica_example.py

# TraCKR Causal Discovery
python cd_trackr_example.py

# LiNGAM Causal Discovery
python cd_lingam_example.py

# Causal Discovery then Decision Analysis
python cd_multica_da_example.py
python cd_trackr_da_example.py
python cd_lingam_da_example.py
```

## Notes

- The examples use async/await syntax
- The SDK uses async context managers for proper resource management
- All API calls are asynchronous and should be awaited
- Error handling is included to demonstrate proper exception handling
- API credentials can be provided via shell environment or `examples/.env`
