"""Shared helpers for examples (API base URL, API key resolution)."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_sdk_test_data_dir() -> Path:
    """Return the SDK test data directory (test_data/ at SDK repo root).

    Returns:
        Path: Path to the test_data directory at SDK repo root.
    """
    return Path(__file__).resolve().parent.parent / "test_data"


def _terraform_output(name: str, terraform_dir: Optional[str] = None) -> str:
    """Get Terraform output value.

    Args:
        name (str): Output name to retrieve
        terraform_dir (Optional[str]): Terraform directory path (defaults to MVP/terraform)

    Returns:
        str: Terraform output value or empty string if not found
    """
    if terraform_dir is None:
        # Get MVP directory (parent of examples)
        mvp_dir = Path(__file__).parent.parent.parent
        terraform_dir = str(mvp_dir / "terraform")

    try:
        result = subprocess.run(
            ["terraform", "output", "-raw", name],
            cwd=str(terraform_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _is_localstack_running() -> bool:
    """Check if LocalStack is running by checking Docker or port.

    Returns:
        bool: True if LocalStack appears to be running
    """
    # Check environment variable first
    endpoint_url = os.getenv("AWS_ENDPOINT_URL", "")
    if endpoint_url and ":4566" in endpoint_url:
        return True

    # Check if LocalStack container is running
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=localstack", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except Exception:
        pass

    # Check if port 4566 is listening (LocalStack default port)
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        port_result = sock.connect_ex(("localhost", 4566))
        sock.close()
        if port_result == 0:
            return True
    except Exception:
        pass

    return False


def _resolve_api_base_url() -> str:
    """Resolve API Gateway base URL from Terraform or environment.

    Returns:
        str: API Gateway base URL or empty string if not available
    """
    # First, try environment variable
    base_url = os.getenv("CAUSAL_AI_BASE_URL")
    if base_url:
        return base_url.rstrip("/")

    # Check if using LocalStack
    endpoint_url = os.getenv("AWS_ENDPOINT_URL", "").rstrip("/")
    is_localstack = _is_localstack_running() or (":4566" in endpoint_url if endpoint_url else False)

    # Get API Gateway ID from Terraform
    api_id = _terraform_output("api_gateway_id")
    if not api_id:
        return ""

    if is_localstack:
        # LocalStack: construct special URL format
        if endpoint_url:
            endpoint_base = endpoint_url.rstrip("/")
            if "://" in endpoint_base:
                localstack_host = endpoint_base.split("://")[1].split(":")[0]
            else:
                localstack_host = "localhost"
        else:
            localstack_host = "localhost"

        # Get stage from api_gateway_url output or use "local" as default
        api_gateway_url = _terraform_output("api_gateway_url")
        if api_gateway_url:
            stage = api_gateway_url.rstrip("/").split("/")[-1] or "local"
        else:
            stage = "local"

        return f"http://{localstack_host}:4566/restapis/{api_id}/{stage}/_user_request_"
    else:
        # AWS: use Terraform output directly
        api_endpoint = _terraform_output("api_gateway_endpoint")
        if api_endpoint:
            return api_endpoint.rstrip("/")

        # Fallback: construct from API Gateway URL
        api_gateway_url = _terraform_output("api_gateway_url")
        if api_gateway_url:
            return api_gateway_url.rstrip("/")

    return ""


def _create_temp_api_key() -> tuple[Optional[str], Optional[str]]:
    """Create temporary API key using cai-keymgr.

    Returns:
        tuple[Optional[str], Optional[str]]: (api_key, company_name) or (None, None) if failed
    """
    try:
        import time

        # Auto-configure LocalStack credentials if LocalStack is detected
        env = os.environ.copy()
        if _is_localstack_running():
            if not env.get("AWS_ENDPOINT_URL"):
                env["AWS_ENDPOINT_URL"] = "http://localhost:4566"
            if not env.get("AWS_ACCESS_KEY_ID"):
                env["AWS_ACCESS_KEY_ID"] = "test"
            if not env.get("AWS_SECRET_ACCESS_KEY"):
                env["AWS_SECRET_ACCESS_KEY"] = "test"

        company_name = f"test-{int(time.time())}-{os.getpid()}"
        result = subprocess.run(
            [
                "cai-keymgr",
                "create",
                "--company",
                company_name,
                "--days",
                "1",
                "--token-balance",
                "20",
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        # Extract API key from output (format: "API Key: cAI_...")
        for line in result.stdout.split("\n"):
            if "API Key:" in line:
                api_key = line.split("API Key:")[1].strip()
                if api_key:
                    logger.info("Created temporary API key for company: %s", company_name)
                    return api_key, company_name
        # Also check stderr in case output goes there
        for line in result.stderr.split("\n"):
            if "API Key:" in line:
                api_key = line.split("API Key:")[1].strip()
                if api_key:
                    logger.info("Created temporary API key for company: %s", company_name)
                    return api_key, company_name
    except FileNotFoundError:
        logger.warning("cai-keymgr not found. Install with: pip install -e MVP/api_key_manager")
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to create API key: %s", e.stderr or e.stdout)

    return None, None


def _delete_api_key(company_name: str) -> None:
    """Delete API key by company name.

    Args:
        company_name (str): Company name associated with the API key
    """
    if not company_name:
        return

    try:
        # Auto-configure LocalStack credentials if LocalStack is detected
        env = os.environ.copy()
        if _is_localstack_running():
            if not env.get("AWS_ENDPOINT_URL"):
                env["AWS_ENDPOINT_URL"] = "http://localhost:4566"
            if not env.get("AWS_ACCESS_KEY_ID"):
                env["AWS_ACCESS_KEY_ID"] = "test"
            if not env.get("AWS_SECRET_ACCESS_KEY"):
                env["AWS_SECRET_ACCESS_KEY"] = "test"

        result = subprocess.run(
            ["cai-keymgr", "delete", "--company", company_name, "--yes"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
            env=env,
        )
        if result.returncode == 0:
            logger.info("Deleted temporary API key for company: %s", company_name)
        else:
            logger.info("API key cleanup skipped for %s (will auto-expire in 1 day)", company_name)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        logger.info("API key cleanup skipped for %s (will auto-expire in 1 day)", company_name)


def _get_api_key() -> tuple[str, Optional[str]]:
    """Get API key from environment or create temporary one.

    Returns:
        tuple[str, Optional[str]]: (api_key, company_name) where company_name is None if from env
    """
    # First, try environment variable
    api_key = os.getenv("CAUSAL_AI_API_KEY")
    if api_key:
        return api_key, None

    # Try to create temporary API key
    api_key, company_name = _create_temp_api_key()
    if api_key:
        return api_key, company_name

    # Fallback: return empty
    logger.error(
        "No API key available. Please set CAUSAL_AI_API_KEY environment variable "
        "or ensure cai-keymgr is installed and working."
    )
    return "", None
