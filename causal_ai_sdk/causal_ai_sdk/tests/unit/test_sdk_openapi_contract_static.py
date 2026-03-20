"""Static SDK/OpenAPI contract checks (no live API calls)."""

from __future__ import annotations

import json
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Type

from causal_ai_sdk.contracts import CONTRACTS  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel

# OpenAPI request-body endpoints not yet in SDK (path, method) — add contract and remove from set
_OPENAPI_REQUEST_BODY_NOT_IN_SDK: Set[Tuple[str, str]] = {
    ("/agent/approve", "post"),
    ("/agent/chat", "post"),
    ("/agent/next", "post"),
    ("/agent/session", "post"),
}


def _schema_props(schema: Dict[str, Any]) -> Set[str]:
    return set((schema.get("properties") or {}).keys())


def _schema_required(schema: Dict[str, Any]) -> Set[str]:
    return set(schema.get("required") or [])


def _unwrap_optional(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return the non-null branch of anyOf [T, null], or the schema itself.

    Args:
        schema (Dict[str, Any]): JSON Schema dict (may have anyOf).

    Returns:
        Dict[str, Any]: Unwrapped schema or original.
    """
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        for branch in any_of:
            if isinstance(branch, dict) and branch.get("type") != "null":
                return _unwrap_optional(branch)
        return schema
    return schema


def _resolve_ref(
    ref: str,
    openapi_schemas: Optional[Dict[str, Dict[str, Any]]],
    defs: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Resolve $ref to a schema dict from OpenAPI components/schemas or Pydantic $defs.

    Args:
        ref (str): $ref value (e.g. #/components/schemas/Foo).
        openapi_schemas (Optional[Dict]): OpenAPI components/schemas.
        defs (Optional[Dict]): Pydantic $defs.

    Returns:
        Optional[Dict[str, Any]]: Resolved schema or None.
    """
    if ref.startswith("#/components/schemas/") and openapi_schemas:
        return openapi_schemas.get(ref.split("/")[-1])
    if (ref.startswith("#/$defs/") or ref.startswith("#/definitions/")) and defs:
        return defs.get(ref.split("/")[-1])
    return None


def _effective_json_type(
    prop_schema: Dict[str, Any],
    *,
    openapi_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
    defs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[str]:
    """Return the effective JSON Schema type, unwrapping anyOf [T, null] and $ref.

    Args:
        prop_schema (Dict[str, Any]): Property schema (type, anyOf, or $ref).
        openapi_schemas (Optional[Dict]): OpenAPI components/schemas.
        defs (Optional[Dict]): Pydantic $defs.

    Returns:
        Optional[str]: One of string, integer, number, boolean, array, object, or None.
    """
    schema = _unwrap_optional(prop_schema)
    ref = schema.get("$ref")
    if isinstance(ref, str):
        resolved = _resolve_ref(ref, openapi_schemas, defs)
        return resolved.get("type") if isinstance(resolved, dict) else None
    return schema.get("type")


def _get_resolved_prop_schema(
    prop_schema: Dict[str, Any],
    *,
    openapi_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
    defs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve property schema to a concrete dict (unwrap optional, follow $ref).

    Args:
        prop_schema (Dict[str, Any]): Property schema.
        openapi_schemas (Optional[Dict]): OpenAPI components/schemas.
        defs (Optional[Dict]): Pydantic $defs.

    Returns:
        Optional[Dict[str, Any]]: Resolved schema dict or None.
    """
    if not isinstance(prop_schema, dict):
        return None
    schema = _unwrap_optional(prop_schema)
    ref = schema.get("$ref")
    if isinstance(ref, str):
        return _resolve_ref(ref, openapi_schemas, defs)
    return schema


def _property_names_type(schema: Dict[str, Any]) -> Optional[str]:
    """Return the type of propertyNames if present (e.g. 'string' for dict key constraint).

    Args:
        schema (Dict[str, Any]): JSON Schema dict (may have propertyNames).

    Returns:
        Optional[str]: propertyNames.type or None.
    """
    pnames = schema.get("propertyNames")
    if isinstance(pnames, dict):
        return pnames.get("type")
    return None


def _resolve_to_object_schema(
    prop_schema: Dict[str, Any],
    *,
    openapi_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
    defs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve to the object schema for recursion (unwrap optional, follow $ref).

    Args:
        prop_schema (Dict[str, Any]): Property schema.
        openapi_schemas (Optional[Dict]): OpenAPI components/schemas.
        defs (Optional[Dict]): Pydantic $defs.

    Returns:
        Optional[Dict[str, Any]]: Object schema with properties, or None.
    """
    resolved = _get_resolved_prop_schema(prop_schema, openapi_schemas=openapi_schemas, defs=defs)
    if not resolved or resolved.get("type") != "object":
        return None
    if "properties" in resolved:
        return resolved
    return None


def _assert_types_compatible(
    openapi_schema: Dict[str, Any],
    model_schema: Dict[str, Any],
    *,
    contract_name: str,
    side: str,
    model_name: str,
    openapi_schemas: Dict[str, Dict[str, Any]],
    root_model_defs: Optional[Dict[str, Dict[str, Any]]] = None,
    depth: int = 10,
    property_path: str = "",
) -> None:
    """Assert compatible JSON Schema types at this level and recurse into nested objects.

    Args:
        openapi_schema (Dict[str, Any]): OpenAPI schema at this level.
        model_schema (Dict[str, Any]): SDK model JSON schema at this level.
        contract_name (str): Contract name for errors.
        side (str): request or response.
        model_name (str): Model name for errors.
        openapi_schemas (Dict): OpenAPI components/schemas.
        root_model_defs (Optional[Dict]): Root $defs for resolving refs.
        depth (int): Max recursion depth.
        property_path (str): Path for error messages.

    Returns:
        None

    Raises:
        AssertionError: If types or shape are incompatible.
    """
    if depth <= 0:
        return
    openapi_props = openapi_schema.get("properties") or {}
    model_props = model_schema.get("properties") or {}
    openapi_required = _schema_required(openapi_schema)
    model_required = _schema_required(model_schema)
    openapi_names = _schema_props(openapi_schema)
    model_names = _schema_props(model_schema)
    defs = root_model_defs if root_model_defs is not None else (model_schema.get("$defs") or {})
    path_msg = f" for property {property_path!r}" if property_path else ""

    missing = openapi_required - model_names
    if missing:
        raise AssertionError(
            f"{contract_name} ({side}): OpenAPI required properties missing from SDK "
            f"schema in {model_name}{path_msg}: {sorted(missing)}"
        )
    wrong_type = model_required - openapi_names
    if wrong_type:
        raise AssertionError(
            f"{contract_name} ({side}): SDK value type{path_msg} does not match OpenAPI: "
            f"the type used for this field has keys {sorted(wrong_type)} that OpenAPI "
            f"does not have here."
        )

    extra_in_sdk = model_names - openapi_names
    if extra_in_sdk:
        raise AssertionError(
            f"{contract_name} ({side}): SDK has properties not in OpenAPI in {model_name}"
            f"{path_msg}: {sorted(extra_in_sdk)}. Remove or align with API schema."
        )

    for prop in openapi_names & model_names:
        openapi_type = _effective_json_type(openapi_props[prop], openapi_schemas=openapi_schemas)
        model_type = _effective_json_type(model_props[prop], defs=defs)
        if openapi_type is None and model_type is None:
            continue
        if openapi_type is None or model_type is None:
            raise AssertionError(
                f"{contract_name} ({side}): property {prop!r} in {model_name}: "
                f"type could not be resolved (OpenAPI={openapi_type!r}, SDK={model_type!r})."
            )
        if openapi_type != model_type:
            raise AssertionError(
                f"{contract_name} ({side}): property {prop!r} type mismatch in "
                f"{model_name}: OpenAPI has {openapi_type!r}, SDK has {model_type!r}"
            )
        if openapi_type == "object" and model_type == "object":
            openapi_resolved = _get_resolved_prop_schema(
                openapi_props[prop], openapi_schemas=openapi_schemas
            )
            model_resolved = _get_resolved_prop_schema(model_props[prop], defs=defs)
            if openapi_resolved is not None and model_resolved is not None:
                openapi_key_type = _property_names_type(openapi_resolved)
                model_key_type = _property_names_type(model_resolved)
                if openapi_key_type is None:
                    openapi_key_type = "string"
                if model_key_type is not None and model_key_type != openapi_key_type:
                    raise AssertionError(
                        f"{contract_name} ({side}): property {prop!r} in {model_name}: "
                        f"dict key type mismatch (OpenAPI expects keys {openapi_key_type!r}, "
                        f"SDK has {model_key_type!r}). Use Dict[str, ...] for string keys."
                    )
            openapi_obj = _resolve_to_object_schema(
                openapi_props[prop], openapi_schemas=openapi_schemas
            )
            model_obj = _resolve_to_object_schema(model_props[prop], defs=defs)
            if openapi_obj is not None and model_obj is not None:
                sub_path = f"{property_path}.{prop}" if property_path else prop
                _assert_types_compatible(
                    openapi_obj,
                    model_obj,
                    contract_name=contract_name,
                    side=side,
                    model_name=model_name,
                    openapi_schemas=openapi_schemas,
                    root_model_defs=defs,
                    depth=depth - 1,
                    property_path=sub_path,
                )


def _dig(data: Any, *keys: str) -> Any:
    """Traverse nested dict with keys; return None if any step is missing or not a dict.

    Args:
        data (Any): Root value (typically a dict) to traverse.
        *keys (str): Sequence of keys to follow in order.

    Returns:
        The value at the end of the key path, or None if any step is missing
        or not a dict.
    """
    return reduce(
        lambda d, k: d.get(k) if isinstance(d, dict) else None,
        keys,
        data,
    )


def _get_openapi_request_schema_ref(
    openapi: Dict[str, Any], path: str, method: str
) -> Optional[str]:
    """Return the request body schema ref name for this path+method, or None.

    Args:
        openapi (Dict[str, Any]): Loaded OpenAPI spec (paths, components).
        path (str): Path string (e.g. /kg/upload-url/{uuid}).
        method (str): HTTP method (e.g. POST).

    Returns:
        The schema name from the requestBody $ref (e.g. KGUploadUrlRequest),
        or None if there is no application/json request body with a $ref.
    """
    ref = _dig(
        openapi,
        "paths",
        path,
        method.lower(),
        "requestBody",
        "content",
        "application/json",
        "schema",
        "$ref",
    )
    if not ref or not ref.startswith("#/components/schemas/"):
        return None
    return ref.split("/")[-1]


def _assert_model_compatible_with_openapi(
    model: Type[BaseModel],
    openapi_schema: Dict[str, Any],
    *,
    contract_name: str,
    side: str,
    allowed_sdk_only_fields: Set[str] | None = None,
) -> None:
    allowed = allowed_sdk_only_fields or set()
    model_schema = model.model_json_schema()
    model_props = _schema_props(model_schema)
    model_required = _schema_required(model_schema)
    openapi_props = _schema_props(openapi_schema)

    missing_in_openapi = (model_props - openapi_props) - allowed
    if missing_in_openapi:
        raise AssertionError(
            f"{contract_name} ({side}): SDK model {model.__name__} has fields not present "
            f"in OpenAPI schema: {sorted(missing_in_openapi)}"
        )

    required_not_declared = model_required - openapi_props
    if required_not_declared:
        raise AssertionError(
            f"{contract_name} ({side}): SDK model {model.__name__} requires fields not present "
            f"in OpenAPI schema: {sorted(required_not_declared)}"
        )


def test_sdk_contract_registry_matches_openapi_shapes():
    """Ensure contract registry models are schema-compatible with openapi.json.

    Raises:
        AssertionError: If OpenAPI paths/schemas drift from SDK contract models.
    """
    repo_root = Path(__file__).resolve().parents[4]

    openapi_path = repo_root / "openapi.json"
    openapi = json.loads(openapi_path.read_text(encoding="utf-8"))
    schemas = openapi["components"]["schemas"]
    paths = openapi["paths"]

    # Request-body endpoint alignment: OpenAPI vs contract registry
    runtime_keys = set()
    for path, path_spec in paths.items():
        for method in ("get", "post", "put", "patch", "delete"):
            if method in path_spec and _get_openapi_request_schema_ref(openapi, path, method):
                runtime_keys.add((path, method))
    contract_keys = {
        (c.path, c.method.lower())
        for c in CONTRACTS
        if c.request_model and c.openapi_request_schema
    }
    missing_contracts = sorted((runtime_keys - contract_keys) - _OPENAPI_REQUEST_BODY_NOT_IN_SDK)
    stale_contracts = sorted(contract_keys - runtime_keys)
    if missing_contracts:
        raise AssertionError(
            "OpenAPI endpoints with request body schema are missing a contract "
            "(request_model + openapi_request_schema): "
            + ", ".join(f"{m.upper()} {p}" for p, m in missing_contracts)
        )
    if stale_contracts:
        raise AssertionError(
            "Contracts declare request body but OpenAPI has no request body schema for: "
            + ", ".join(f"{m.upper()} {p}" for p, m in stale_contracts)
        )

    for contract in CONTRACTS:
        for field_name in contract.request_sdk_only_fields:
            reason = contract.request_sdk_only_reasons.get(field_name, "").strip()
            if not reason:
                raise AssertionError(
                    f"{contract.name} (request): SDK-only field '{field_name}' must include "
                    "a non-empty reason."
                )
        for field_name in contract.response_sdk_only_fields:
            reason = contract.response_sdk_only_reasons.get(field_name, "").strip()
            if not reason:
                raise AssertionError(
                    f"{contract.name} (response): SDK-only field '{field_name}' must include "
                    "a non-empty reason."
                )

        if contract.path not in paths:
            raise AssertionError(f"{contract.name}: path not found in OpenAPI ({contract.path})")
        if contract.method.lower() not in paths[contract.path]:
            raise AssertionError(
                f"{contract.name}: method not found in OpenAPI "
                f"({contract.method} {contract.path})"
            )

        if contract.request_model and contract.openapi_request_schema:
            openapi_ref = _get_openapi_request_schema_ref(openapi, contract.path, contract.method)
            if openapi_ref != contract.openapi_request_schema:
                raise AssertionError(
                    f"{contract.name}: contract openapi_request_schema "
                    f"{contract.openapi_request_schema!r} does not match OpenAPI "
                    f"request body schema {openapi_ref!r} for {contract.method} {contract.path}"
                )
            if contract.openapi_request_schema not in schemas:
                raise AssertionError(
                    f"{contract.name}: request schema missing from OpenAPI: "
                    f"{contract.openapi_request_schema}"
                )
            _assert_model_compatible_with_openapi(
                contract.request_model,
                schemas[contract.openapi_request_schema],
                contract_name=contract.name,
                side="request",
                allowed_sdk_only_fields=set(contract.request_sdk_only_fields),
            )
            openapi_req_schema = schemas[contract.openapi_request_schema]
            openapi_required = _schema_required(openapi_req_schema)
            if openapi_required:
                model_schema = contract.request_model.model_json_schema()
                model_props = _schema_props(model_schema)
                model_required = _schema_required(model_schema)
                missing_in_sdk = openapi_required - model_props
                if missing_in_sdk:
                    raise AssertionError(
                        f"{contract.name} (request): OpenAPI required fields missing from SDK "
                        f"model {contract.request_model.__name__}: {sorted(missing_in_sdk)}"
                    )
                optional_in_sdk = (openapi_required & model_props) - model_required
                if optional_in_sdk:
                    raise AssertionError(
                        f"{contract.name} (request): OpenAPI required fields must be required "
                        f"in SDK model {contract.request_model.__name__}: {sorted(optional_in_sdk)}"
                    )
            _req_schema = contract.request_model.model_json_schema()
            _assert_types_compatible(
                openapi_req_schema,
                _req_schema,
                contract_name=contract.name,
                side="request",
                model_name=contract.request_model.__name__,
                openapi_schemas=schemas,
                root_model_defs=_req_schema.get("$defs"),
            )

        if contract.response_model and contract.openapi_response_schema:
            if contract.openapi_response_schema not in schemas:
                raise AssertionError(
                    f"{contract.name}: response schema missing from OpenAPI: "
                    f"{contract.openapi_response_schema}"
                )
            _assert_model_compatible_with_openapi(
                contract.response_model,
                schemas[contract.openapi_response_schema],
                contract_name=contract.name,
                side="response",
                allowed_sdk_only_fields=set(contract.response_sdk_only_fields),
            )
            _resp_schema = contract.response_model.model_json_schema()
            _assert_types_compatible(
                schemas[contract.openapi_response_schema],
                _resp_schema,
                contract_name=contract.name,
                side="response",
                model_name=contract.response_model.__name__,
                openapi_schemas=schemas,
                root_model_defs=_resp_schema.get("$defs"),
            )
