"""httpx library implementation of async HTTP client."""

import json
import warnings
from typing import Dict, Optional

import httpx
from causal_ai_sdk.exceptions import (
    APIError,
    AuthenticationError,
    NetworkError,
    NotFoundError,
    ValidationError,
)
from causal_ai_sdk.http.client import HTTPClient


class HttpxHTTPClient(HTTPClient):
    """Async HTTP client implementation using httpx library.

    Must be used as an async context manager. Do not call methods without
    ``async with HttpxHTTPClient(...) as client:``.
    """

    def __init__(self, timeout: int = 120):
        """Initialize async HTTP client.

        Args:
            timeout (int): Default request timeout in seconds.
        """
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry.

        Returns:
            HttpxHTTPClient: Self instance
        """
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ):
        """Async context manager exit.

        Args:
            exc_type (Optional[type[BaseException]]): Exception type if exception occurred
            exc_val (Optional[BaseException]): Exception value if exception occurred
            exc_tb (Optional[object]): Exception traceback if exception occurred

        Closes the HTTP client.
        """
        await self.aclose()

    def __del__(self):
        """Warn if the client was never closed (e.g. context manager not exited)."""
        if self._client is not None:
            warnings.warn(
                "HttpxHTTPClient was not closed. Use 'async with' and ensure the block exits.",
                ResourceWarning,
                source=self,
                stacklevel=1,
            )

    async def aclose(self):
        """Close the HTTP client. Called automatically by the context manager exit."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> Dict:
        """Send async HTTP request using httpx library.

        Args:
            method (str): HTTP method (GET, POST, PUT, DELETE, etc.)
            url (str): Request URL
            headers (Optional[Dict[str, str]]): Request headers
            params (Optional[Dict]): Query parameters
            json_data (Optional[Dict]): JSON request body
            timeout (Optional[int]): Request timeout in seconds (overrides default)

        Returns:
            Dict: Response data (parsed JSON)

        Raises:
            RuntimeError: If the client is not used as async context manager.
            NetworkError: For network-related errors
        """
        if self._client is None:
            raise RuntimeError(
                "HttpxHTTPClient must be used as async context manager. "
                "Use: async with HttpxHTTPClient(...) as client:"
            )

        headers = headers or {}

        try:
            response = await self._client.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=timeout or self.timeout,
            )

            # Handle HTTP errors
            return self._handle_response(response)

        except httpx.TimeoutException as e:
            raise NetworkError(f"Request timeout: {str(e)}") from e
        except httpx.ConnectError as e:
            raise NetworkError(f"Connection error: {str(e)}") from e
        except httpx.NetworkError as e:
            raise NetworkError(f"Network error: {str(e)}") from e
        except httpx.HTTPError as e:
            raise NetworkError(f"HTTP error: {str(e)}") from e

    def _handle_response(self, response: httpx.Response) -> Dict:
        """Handle HTTP response and convert to appropriate exception if needed.

        Args:
            response (httpx.Response): httpx.Response object

        Returns:
            Dict: Parsed JSON response data

        Raises:
            AuthenticationError: For 401/403 status codes
            NotFoundError: For 404 status code
            ValidationError: For 400 status code
            APIError: For other error status codes
        """
        status_code = response.status_code

        # Try to parse response as JSON
        try:
            response_data = response.json() if response.content else {}
        except (ValueError, json.JSONDecodeError):
            response_data = {"message": response.text or "Unknown error"}

        # Prefer "message" (e.g. Knowledge Graph detail), then "error" (e.g. Causal Discovery)
        def _error_message(fallback: str) -> str:
            return response_data.get("message") or response_data.get("error") or fallback

        # Handle error status codes (401/403, 404, 400 handled first; else = other 4xx/5xx)
        if status_code in (401, 403):
            raise AuthenticationError(
                message=_error_message("Authentication failed"),
                status_code=status_code,
                response_data=response_data,
            )
        elif status_code == 404:
            raise NotFoundError(
                message=_error_message("Resource not found"),
                status_code=status_code,
                response_data=response_data,
            )
        elif status_code == 400:
            raise ValidationError(
                message=_error_message("Validation error"),
                status_code=status_code,
                response_data=response_data,
            )
        elif status_code >= 400:
            raise APIError(
                message=_error_message(f"API error: {status_code}"),
                status_code=status_code,
                response_data=response_data,
            )

        # Success - return parsed JSON
        return response_data
