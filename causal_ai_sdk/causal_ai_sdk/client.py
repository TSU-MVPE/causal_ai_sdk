"""Main client entry point."""

from types import TracebackType
from typing import Optional, Type

from causal_ai_sdk.config import Config
from causal_ai_sdk.http.client import HTTPClient
from causal_ai_sdk.http.httpx_client import HttpxHTTPClient
from causal_ai_sdk.services.cd_service import CDService
from causal_ai_sdk.services.da_service import DAService
from causal_ai_sdk.services.kg_service import KGService


class CausalAIClient:
    """Main client for Causal AI SDK.

    This client provides a unified interface to access all services.
    It implements the Facade pattern, hiding the complexity of service
    initialization and configuration.

    The client must be used as an async context manager. Do not use without
    ``async with CausalAIClient(...) as client:``.

    Example:
        >>> async with CausalAIClient(
        ...     api_key="your-api-key",
        ...     base_url="https://api.example.com"
        ... ) as client:
        ...     session = await client.kg.init_session()
        ...     kg = await client.kg.upload_kg_from_file(
        ...         session_uuid=session.uuid,
        ...         file_path="graph.json"
        ...     )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        http_client: Optional[HTTPClient] = None,
        config: Optional[Config] = None,
    ):
        """Initialize Causal AI client.

        Args:
            api_key (str): API Key for authentication
            base_url (str): Base URL of the API Gateway
            http_client (Optional[HTTPClient]): Optional HTTP client instance
                (defaults to HttpxHTTPClient). When provided, the caller is
                responsible for its lifecycle (use as async context manager or
                close it when done).
            config (Optional[Config]): Optional Config instance
                (will be created if not provided)
        """
        # Create or use provided Config
        self.config = config or Config(api_key=api_key, base_url=base_url)

        # Create or use provided HTTP client
        if http_client is None:
            self._http_client: HTTPClient = HttpxHTTPClient(timeout=self.config.timeout)
            self._owns_http_client = True
        else:
            self._http_client = http_client
            self._owns_http_client = False

        # Initialize services
        self._kg_service = KGService(self.config, self._http_client)
        self._cd_service = CDService(self.config, self._http_client)
        self._da_service = DAService(self.config, self._http_client)

    async def __aenter__(self):
        """Async context manager entry.

        Returns:
            CausalAIClient: Self instance
        """
        # Enter HTTP client context if it supports async context manager
        if hasattr(self._http_client, "__aenter__"):
            await self._http_client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ):
        """Async context manager exit.

        Closes the HTTP client if we own it.

        Args:
            exc_type (Optional[Type[BaseException]]): Exception type
            exc_val (Optional[BaseException]): Exception value
            exc_tb (Optional[TracebackType]): Exception traceback
        """
        # Exit HTTP client context if it supports async context manager
        if self._owns_http_client and hasattr(self._http_client, "__aexit__"):
            await self._http_client.__aexit__(exc_type, exc_val, exc_tb)
        elif self._owns_http_client and hasattr(self._http_client, "aclose"):
            await self._http_client.aclose()

    @property
    def kg(self) -> KGService:
        """Knowledge Graph service.

        Returns:
            KGService: Knowledge Graph service instance
        """
        return self._kg_service

    @property
    def cd(self) -> CDService:
        """Causal Discovery service.

        Returns:
            CDService: Causal Discovery service instance
        """
        return self._cd_service

    @property
    def da(self) -> DAService:
        """Decision Analysis service.

        Returns:
            DAService: Decision Analysis service instance.
        """
        return self._da_service
