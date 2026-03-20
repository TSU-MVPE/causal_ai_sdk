"""Exception definitions for Causal AI SDK."""

from typing import Dict, Optional


class CausalAIError(Exception):
    """Base exception class for all Causal AI SDK errors.

    All SDK-specific exceptions inherit from this class, allowing users
    to catch all SDK errors with a single exception handler.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict] = None,
    ):
        """Initialize exception.

        Args:
            message (str): Error message.
            status_code (Optional[int]): HTTP status code (if applicable).
            response_data (Optional[Dict]): Response data from API (if applicable).
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data

    def __str__(self) -> str:
        """String representation of exception.

        Returns:
            String representation of the exception.
        """
        if self.status_code:
            return f"{self.message} (Status: {self.status_code})"
        return self.message


class APIError(CausalAIError):
    """Exception raised for API errors.

    This exception is raised when the API returns an error response
    that is not covered by more specific exception types.
    """

    pass


class AuthenticationError(CausalAIError):
    """Exception raised for authentication errors.

    This exception is raised when:
    - API key is missing
    - API key is invalid
    - API key has expired
    - User lacks permission to access the resource
    """

    pass


class NotFoundError(CausalAIError):
    """Exception raised when a requested resource is not found.

    This exception is raised when:
    - Session UUID does not exist
    - Knowledge graph ID does not exist
    - Task ID does not exist
    """

    pass


class ValidationError(CausalAIError):
    """Exception raised for validation errors.

    This exception is raised when:
    - Request parameters are invalid
    - Request body format is incorrect
    - Required fields are missing
    """

    pass


class NetworkError(CausalAIError):
    """Exception raised for network-related errors.

    This exception is raised when:
    - Connection timeout
    - Connection refused
    - DNS resolution failure
    - Other network-related issues
    """

    pass
