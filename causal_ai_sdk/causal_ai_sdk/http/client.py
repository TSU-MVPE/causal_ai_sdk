"""HTTP client abstract interface."""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class HTTPClient(ABC):
    """Abstract interface for async HTTP clients.

    This interface allows for different HTTP client implementations
    (e.g., httpx) to be used interchangeably.
    """

    @abstractmethod
    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> Dict:
        """Send async HTTP request.

        Args:
            method (str): HTTP method (GET, POST, PUT, DELETE, etc.)
            url (str): Request URL
            headers (Optional[Dict[str, str]]): Request headers
            params (Optional[Dict]): Query parameters
            json_data (Optional[Dict]): JSON request body
            timeout (Optional[int]): Request timeout in seconds

        Returns:
            Dict: Response data (parsed JSON)

        Raises:
            NetworkError: For network-related errors
            APIError: For API errors
        """
        pass
