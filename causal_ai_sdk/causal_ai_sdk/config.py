"""Configuration management."""

import os
from typing import Any, Optional
from urllib.parse import urlparse


class Config:
    """Configuration class for Causal AI SDK.

    Manages API key, base URL, and other configuration settings.
    Supports reading from environment variables.

    Args:
        api_key: API Key for authentication. If not provided, will try to read
            from CAUSAL_AI_API_KEY environment variable.
        base_url: Base URL of the API Gateway. If not provided, will try to read
            from CAUSAL_AI_BASE_URL environment variable.
        timeout: Request timeout in seconds. Defaults to 120.
        **kwargs: Additional configuration options.

    Raises:
        ValueError: If required configuration is missing or invalid.

    Example:
        >>> # From parameters
        >>> config = Config(
        ...     api_key="your-api-key",
        ...     base_url="https://api.example.com"
        ... )
        >>>
        >>> # From environment variables
        >>> # Set CAUSAL_AI_API_KEY and CAUSAL_AI_BASE_URL
        >>> config = Config()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        **kwargs: Any,
    ):
        """Initialize Config instance.

        Args:
            api_key (Optional[str]): API Key for authentication.
            base_url (Optional[str]): Base URL of the API Gateway.
            timeout (int): Request timeout in seconds.
            **kwargs (Any): Additional configuration options.
        """
        # Read from environment variables if not provided
        self.api_key = api_key or os.getenv("CAUSAL_AI_API_KEY")
        self.base_url = base_url or os.getenv("CAUSAL_AI_BASE_URL")

        # Additional configuration
        self.timeout = timeout
        self._extra = kwargs

        # Validate configuration
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not self.api_key:
            raise ValueError(
                "API key is required. Provide it as a parameter or set "
                "CAUSAL_AI_API_KEY environment variable."
            )

        if not self.base_url:
            raise ValueError(
                "Base URL is required. Provide it as a parameter or set "
                "CAUSAL_AI_BASE_URL environment variable."
            )

        # Validate base_url format
        try:
            parsed = urlparse(self.base_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(
                    f"Invalid base_url format: {self.base_url}. "
                    "Expected format: https://api.example.com"
                )
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"Base URL must use http or https scheme, got: {parsed.scheme}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Invalid base_url format: {self.base_url}") from e

        # Remove trailing slash from base_url
        self.base_url = self.base_url.rstrip("/")

        # Validate timeout
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive, got: {self.timeout}")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get additional configuration value.

        Args:
            key (str): Configuration key.
            default (Optional[Any]): Default value if key is not found.

        Returns:
            Any: Configuration value or default.
        """
        return self._extra.get(key, default)

    def __repr__(self) -> str:
        """String representation of Config.

        Returns:
            String representation of the Config object.
        """
        # Mask API key for security
        masked_key = f"{self.api_key[:8]}..." if self.api_key else None
        return (
            f"Config(api_key={masked_key!r}, base_url={self.base_url!r}, "
            f"timeout={self.timeout})"
        )
