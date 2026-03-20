"""Base service class."""

from typing import Dict, Optional

from causal_ai_sdk.config import Config
from causal_ai_sdk.http.client import HTTPClient


class BaseService:
    """Base service class providing common functionality for all services.

    This class implements the Template Method pattern, providing a common
    structure for making HTTP requests while allowing subclasses to customize
    specific behavior.

    Attributes:
        config: Configuration instance
        http_client: HTTP client instance for making requests
    """

    def __init__(self, config: Config, http_client: HTTPClient):
        """Initialize base service.

        Args:
            config (Config): Configuration instance
            http_client (HTTPClient): HTTP client instance
        """
        self.config = config
        self.http_client = http_client

    def _build_url(self, path: str) -> str:
        """Build full URL from base URL and path.

        Args:
            path (str): API path (e.g., "/kg/init")

        Returns:
            str: Full URL string
        """
        # Ensure path starts with /
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.config.base_url}{path}"

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers with API key.

        Returns:
            Dictionary of headers
        """
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        return headers

    async def _make_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> Dict:
        """Send async HTTP request (Template Method).

        This method implements the template method pattern:
        1. Build full URL
        2. Build headers
        3. Send async HTTP request
        4. Return response data

        Args:
            method (str): HTTP method (GET, POST, PUT, DELETE, etc.)
            path (str): API path (e.g., "/kg/init")
            params (Optional[Dict]): Query parameters
            json_data (Optional[Dict]): JSON request body
            timeout (Optional[int]): Request timeout in seconds (optional)

        Returns:
            Dict: Response data (parsed JSON dictionary)
        """
        # Build URL and headers
        url = self._build_url(path)
        headers = self._build_headers()

        # Send async HTTP request
        response = await self.http_client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json_data=json_data,
            timeout=timeout or self.config.timeout,
        )

        return response
