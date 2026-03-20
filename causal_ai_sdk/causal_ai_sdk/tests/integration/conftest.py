"""Shared fixtures for SDK integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers.timeout import (
    get_polling_timeout_seconds,
    get_request_timeout_seconds,
)


def _sdk_base_url(api_base_url: str) -> str:
    """Strip /kg from tests' api_base_url to get API gateway root for SDK.

    Args:
        api_base_url (str): KG API base URL (e.g. with /kg suffix).

    Returns:
        Base URL without /kg for CausalAIClient.
    """
    return api_base_url.replace("/kg", "").rstrip("/")


def _sdk_test_data_dir() -> Path:
    """Return SDK test_data directory (causal_ai_sdk/test_data).

    Returns:
        Path to causal_ai_sdk/test_data.
    """
    return Path(__file__).resolve().parents[3] / "test_data"


@pytest.fixture
def sdk_request_timeout() -> int:
    """HTTP request timeout in seconds (from tests.helpers.timeout).

    Returns:
        int: Timeout in seconds.
    """
    return get_request_timeout_seconds()


@pytest.fixture
def sdk_polling_timeout() -> int:
    """Polling/wait timeout in seconds (from tests.helpers.timeout).

    Returns:
        int: Timeout in seconds.
    """
    return get_polling_timeout_seconds()


@pytest.fixture
def sdk_base_url(api_base_url):
    """API base URL without /kg for CausalAIClient.

    Args:
        api_base_url (str): From MVP tests.conftest.

    Returns:
        Base URL for CausalAIClient.
    """
    return _sdk_base_url(api_base_url)


@pytest.fixture
def sdk_test_data_dir():
    """Path to causal_ai_sdk/test_data (synced with MVP/scripts/test_data).

    Returns:
        Path to test_data directory.
    """
    return _sdk_test_data_dir()
