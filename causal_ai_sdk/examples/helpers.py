"""Shared helpers for examples (environment-based config)."""

import os
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent
_DOTENV_LOADED = False


def get_sdk_test_data_dir() -> Path:
    """Return the SDK test data directory.

    Returns:
        Path: The ``test_data`` directory at SDK repository root.
    """
    return Path(__file__).resolve().parent.parent / "test_data"


def get_examples_env_path() -> Path:
    """Return the .env path used by SDK examples.

    Returns:
        Path: Absolute path to ``MVP/causal_ai_sdk/examples/.env``.
    """
    return _EXAMPLES_DIR / ".env"


def load_examples_dotenv(override: bool = False) -> None:
    """Load examples .env values into process environment.

    Args:
        override (bool): Whether values from ``.env`` should override existing
            environment variables.

    Raises:
        ImportError: If ``python-dotenv`` is not installed when ``.env`` exists.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED and not override:
        return

    env_path = get_examples_env_path()
    if not env_path.is_file():
        _DOTENV_LOADED = True
        return

    try:
        from dotenv import load_dotenv  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "python-dotenv is required to load examples/.env. "
            "Install it with: pip install python-dotenv"
        ) from exc

    load_dotenv(dotenv_path=env_path, override=override)

    _DOTENV_LOADED = True


def get_api_key_from_env() -> str:
    """Return API key from environment/.env.

    Returns:
        str: API key value, or empty string when not set.
    """
    load_examples_dotenv()
    return os.getenv("CAUSAL_AI_API_KEY", "")


def get_base_url_from_env() -> str:
    """Return API base URL from environment/.env.

    Returns:
        str: Base URL with trailing slash removed, or empty string when not set.
    """
    load_examples_dotenv()
    base_url = os.getenv("CAUSAL_AI_BASE_URL")
    return base_url.rstrip("/") if base_url else ""
