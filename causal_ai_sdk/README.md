# Causal AI SDK

Python SDK for the Causal AI Platform, providing a simple and intuitive interface to interact with Knowledge Graph and Causal Discovery services.

## Features

- **Knowledge Graph Service**: Create, manage, and query causal knowledge graphs
- **Causal Discovery Service**: Run MultiCa, TraCKR, and LiNGAM algorithms; column matching for MultiCa and TraCKR
- **Decision Analysis Service**: Explain and enumerate tasks over a completed CD result
- **File Format Support**: Datasets (CD) and knowledge graphs: CSV and JSON.
- **Type Safety**: Full type hints for better IDE support and code reliability
- **Error Handling**: Comprehensive exception hierarchy for clear error messages

## Installation

### Basic Installation

```bash
pip install causal-ai-sdk
```

### Development Installation

For development, install with all dependencies including testing tools:

```bash
pip install causal-ai-sdk[dev]
```

Or install from source:

```bash
cd causal_ai_sdk
pip install -e ".[dev]"
```

The `[dev]` extra includes testing tools: `pytest`, `pytest-asyncio`, `pytest-cov`, `respx`.

## Quick Start

The client **must** be used as an async context manager. All API calls go inside `async with CausalAIClient(...) as client:`.

### Initialize the Client and Knowledge Graph Service

```python
import asyncio
from causal_ai_sdk import CausalAIClient


async def main():
    async with CausalAIClient(
        api_key="your-api-key",
        base_url="https://api.example.com"
    ) as client:
        # Initialize a session
        session = await client.kg.init_session()
        session_uuid = session["uuid"]
        print(f"Session UUID: {session_uuid}")

        # Upload a KG file (JSON/CSV)
        kg = await client.kg.upload_kg_from_file(
            session_uuid=session_uuid,
            file_path="graph.json",
            title="My Knowledge Graph",
        )

        # List all knowledge graphs in the session
        kg_list = await client.kg.list_kg(session_uuid)

        # Get a specific knowledge graph
        kg_detail = await client.kg.get_kg(session_uuid, kg_id=kg["id"])


if __name__ == "__main__":
    asyncio.run(main())
```

### Causal Discovery Service (MultiCa)

MultiCa requires uploading data, starting column matching, waiting for it to complete, then running discovery. The matching task ID is **required** for `run_multica`:

```python
async with CausalAIClient(api_key="...", base_url="...") as client:
    session = await client.kg.init_session()
    # Upload multiple datasets (e.g. two CSV files)
    uploaded = await client.cd.upload_data_for_multica(
        session["uuid"], ["dataset1.csv", "dataset2.csv"]
    )
    # Start matching and get the required matching_task_id
    matching = await client.cd.start_multica_matching(uploaded)
    matching_task_id = matching["matching_task_id"]
    await client.cd.wait_for_matching(
        session["uuid"], mode="multica",
        matching_task_id=matching_task_id, timeout=300, interval=3
    )
    # matching_task_id is required
    task = await client.cd.run_multica(
        uploaded, matching_task_id=matching_task_id, threshold=0.01
    )
    await client.cd.wait_for_task(
        task["task_id"], session_uuid=session["uuid"], timeout=300, interval=5
    )
    result = await client.cd.get_task_result(session["uuid"], task["task_id"])
    print(f"Result URL: {result['result_url']}")
```

### Decision Analysis (after a successful CD run)

DA references the CD job via `cd_result_reference`. 

```python
async with CausalAIClient(api_key="...", base_url="...") as client:
    da_task = await client.da.run_explain(
        session_uuid=cd_session_uuid,
        cd_result_reference={
            "session_uuid": cd_session_uuid,
            "task_id": cd_task_id,
        },
        current_observation={"debt": 0.4, "age": 35.0},
        targets=[{"col": "debt", "sense": ">", "threshold": 0.8}],
        params={"alpha": 1.0, "time_limit": 60.0},
    )
    await client.da.wait_for_task(cd_session_uuid, da_task["task_id"], timeout=300, interval=5)
    da_result = await client.da.get_task_result(cd_session_uuid, da_task["task_id"])
```

## Examples

See [examples/](examples/) for end-to-end scripts and Jupyter notebooks.

## Contract Layer and Guardrails

The SDK includes a `contracts` layer that serves two purposes:

- **Runtime consistency**: core service calls build paths and validate request payloads through shared contract definitions.
- **Static guardrail**: a fast test compares SDK contract models with the OpenAPI snapshot used by
  the SDK tests to catch drift before runtime.

### Quick local check

Run this from the `causal_ai_sdk` directory:

```bash
pytest tests/api/test_sdk_openapi_contract_static.py -q
```

### SDK-only allowlist rule

If a contract intentionally contains SDK-only fields not present in OpenAPI, those fields must be
explicitly allowlisted **with a non-empty reason**. The static test fails when allowlisted fields
do not have reasons.

## Adjacency Orientation Contract

- Knowledge Graph matrix uploads (`data`) use `cause=row, effect=col`.
- Causal Discovery result `adjacency_matrices` use the same `cause=row, effect=col` orientation.
- Semantics: `adjacency_matrices[i][j] != 0` means `feature_names[i]` causes `feature_names[j]`.
- Decision Analysis internally converts CD adjacency to Cation's required
  `effect=row, cause=col` orientation during preprocessing.

## Requirements

- Python 3.10 or higher

Required dependencies (declared in `pyproject.toml`; installed automatically with the SDK):

- **httpx** — HTTP client for API requests
- **pydantic** — Data validation and settings
- **pandas** — Data handling and file conversion (CSV, etc.)
- **openpyxl** — Excel file support

Optional development dependencies (install with `pip install causal-ai-sdk[dev]`):

- **pytest**, **pytest-asyncio**, **pytest-cov** — Testing
- **respx** — HTTP mocking for tests
