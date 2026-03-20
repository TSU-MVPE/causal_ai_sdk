"""Typed contract metadata used by SDK/OpenAPI static checks."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type

from pydantic import BaseModel


@dataclass(frozen=True)
class EndpointContract:
    """Contract mapping between an SDK endpoint and OpenAPI schemas."""

    name: str
    method: str
    path: str
    request_model: Optional[Type[BaseModel]] = None
    response_model: Optional[Type[BaseModel]] = None
    openapi_request_schema: Optional[str] = None
    openapi_response_schema: Optional[str] = None
    request_sdk_only_fields: Tuple[str, ...] = ()
    response_sdk_only_fields: Tuple[str, ...] = ()
    request_sdk_only_reasons: Dict[str, str] = field(default_factory=dict)
    response_sdk_only_reasons: Dict[str, str] = field(default_factory=dict)
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Ensure SDK-only allowlist fields always provide explicit reasons.

        Raises:
            ValueError: If SDK-only fields and reason-map keys do not match exactly.
        """
        req_fields = set(self.request_sdk_only_fields)
        req_reasons = set(self.request_sdk_only_reasons.keys())
        if req_fields != req_reasons:
            raise ValueError(
                f"{self.name}: request_sdk_only_fields and request_sdk_only_reasons keys must "
                f"match exactly. fields={sorted(req_fields)}, reasons={sorted(req_reasons)}"
            )

        resp_fields = set(self.response_sdk_only_fields)
        resp_reasons = set(self.response_sdk_only_reasons.keys())
        if resp_fields != resp_reasons:
            raise ValueError(
                f"{self.name}: response_sdk_only_fields and response_sdk_only_reasons keys must "
                f"match exactly. fields={sorted(resp_fields)}, reasons={sorted(resp_reasons)}"
            )
