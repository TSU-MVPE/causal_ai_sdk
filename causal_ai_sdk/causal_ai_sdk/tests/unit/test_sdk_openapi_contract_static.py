"""Static SDK/OpenAPI contract checks (no live API calls)."""

from __future__ import annotations

import json
import sys
from functools import reduce
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

from causal_ai_sdk.contracts import CONTRACTS  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel, StrictStr

_JSON_SCALAR_TYPES = frozenset({"string", "integer", "number", "boolean", "null"})


def _hint_unresolved_json_schema_type(
    openapi_type: Optional[str],
    sdk_type: Optional[str],
    property_path: str,
) -> str:
    """Return a short hint when one side has no resolved JSON Schema ``type``.

    Args:
        openapi_type (Optional[str]): OpenAPI JSON type string, if resolved.
        sdk_type (Optional[str]): SDK JSON type string, if resolved.
        property_path (str): Field path; may include ``[*]`` for list elements.

    Returns:
        str: Text to append to an assertion message.
    """
    if sdk_type is None and openapi_type is not None:
        if "[*]" in property_path:
            return (
                f" SDK list elements have no JSON Schema type (often List[Any]); OpenAPI "
                f"expects {openapi_type!r} items—use a concrete element type such as "
                f"List[str] or List[StrictStr]."
            )
        return (
            " SDK schema has no JSON type here (Any, composition without type, etc.); "
            "align the field with OpenAPI."
        )
    if openapi_type is None and sdk_type is not None:
        return " OpenAPI has no single resolved JSON type here; check $ref/oneOf in the spec."
    return " Neither side resolved to a single JSON type; check $ref and composition."


def _raise_unresolved_json_type(
    contract_name: str,
    side: str,
    where: str,
    openapi_type: Optional[str],
    sdk_type: Optional[str],
    hint_path: str,
) -> None:
    """Raise ``AssertionError`` for a missing JSON Schema type on one side.

    Args:
        contract_name (str): Contract label.
        side (str): ``request`` or ``response``.
        where (str): Short description of the schema location.
        openapi_type (Optional[str]): OpenAPI type, if any.
        sdk_type (Optional[str]): SDK type, if any.
        hint_path (str): Path for hint text (e.g. may include ``[*]``).

    Raises:
        AssertionError: Always.
    """
    hint = _hint_unresolved_json_schema_type(openapi_type, sdk_type, hint_path)
    raise AssertionError(
        f"{contract_name} ({side}): {where}: "
        f"type could not be resolved (OpenAPI={openapi_type!r}, SDK={sdk_type!r}).{hint}"
    )


# OpenAPI request-body endpoints not yet in SDK (path, method) — add contract and remove from set
_OPENAPI_REQUEST_BODY_NOT_IN_SDK: Set[Tuple[str, str]] = {
    ("/agent/approve", "post"),
    ("/agent/chat", "post"),
    ("/agent/next", "post"),
    ("/agent/session", "post"),
}


def _unwrap_annotation(ann: Any) -> Any:
    """Strip single-branch ``Optional`` / ``Union[T, None]`` from a type annotation.

    Args:
        ann (Any): A typing annotation (possibly optional/union).

    Returns:
        Any: The inner non-``None`` type, or ``ann`` unchanged.
    """
    if ann is None:
        return ann
    origin = get_origin(ann)
    args = get_args(ann)
    if origin is Union:
        non_none = [a for a in args if a is not type(None) and a is not None]
        if len(non_none) == 1:
            return _unwrap_annotation(non_none[0])
    if sys.version_info >= (3, 10):
        import types

        if origin is types.UnionType:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return _unwrap_annotation(non_none[0])
    return ann


def _python_dict_slot_json_type(ann: Any, slot: int) -> Optional[str]:
    """Map ``Dict`` key (0) or value (1) to a JSON Schema type name.

    Args:
        ann (Any): Annotation for ``Dict[K, V]`` (after optional unwrap).
        slot (int): ``0`` for key type, ``1`` for value type.

    Returns:
        Optional[str]: JSON type string, sentinels for ``typing.Any``, or None.
    """
    ann = _unwrap_annotation(ann)
    origin = get_origin(ann)
    if origin not in (dict, Dict):
        return None
    args = get_args(ann)
    if len(args) < 2:
        return None
    part = _unwrap_annotation(args[slot])
    if part is Any:
        return "__any_key__" if slot == 0 else "__any_value__"
    if part is str or part is StrictStr:
        return "string"
    if part is int:
        return "integer"
    if part is float:
        return "number"
    if part is bool:
        return "boolean"
    return None


def _assert_python_dict_annotation_vs_openapi_object(
    dict_ann: Any,
    openapi_object_resolved: Dict[str, Any],
    *,
    contract_name: str,
    side: str,
    model_name: str,
    path_msg: str,
    field_name: str,
    openapi_schemas: Dict[str, Dict[str, Any]],
    defs: Dict[str, Dict[str, Any]],
) -> None:
    """Compare a ``Dict[K, V]`` annotation to an OpenAPI object schema.

    Args:
        dict_ann (Any): The ``Dict[...]`` annotation for the field or list element.
        openapi_object_resolved (Dict[str, Any]): Resolved OpenAPI JSON Schema object.
        contract_name (str): Contract label for errors.
        side (str): ``request`` or ``response``.
        model_name (str): Pydantic model name for errors.
        path_msg (str): Suffix describing the property path.
        field_name (str): Python field name for errors.
        openapi_schemas (Dict[str, Dict[str, Any]]): OpenAPI ``components/schemas``.
        defs (Dict[str, Dict[str, Any]]): Pydantic ``$defs``.

    Raises:
        AssertionError: If key or value typing disagrees with OpenAPI.
    """
    py_key = _python_dict_slot_json_type(dict_ann, 0)
    if py_key == "__any_key__":
        raise AssertionError(
            f"{contract_name} ({side}): dict key typing too loose in {model_name}{path_msg}: "
            f"field {field_name!r} uses ``Dict[Any, …]``. Use ``Dict[str, …]`` (JSON object "
            f"keys are always strings)."
        )
    if py_key is not None and py_key != "string":
        raise AssertionError(
            f"{contract_name} ({side}): dict key type mismatch in {model_name}{path_msg}: "
            f"JSON object keys are strings; Python field {field_name!r} uses {py_key!r} keys. "
            f"Use ``Dict[str, …]``."
        )

    ap = openapi_object_resolved.get("additionalProperties")
    if ap is True or not isinstance(ap, dict):
        return
    o_val = _effective_json_type(ap, openapi_schemas=openapi_schemas, defs=defs)
    if o_val is None:
        return
    py_val = _python_dict_slot_json_type(dict_ann, 1)
    if py_val is None:
        return
    if py_val == "__any_value__":
        raise AssertionError(
            f"{contract_name} ({side}): dict value typing too loose in {model_name}{path_msg}: "
            f"field {field_name!r} uses ``Dict[..., Any]`` but OpenAPI constrains map values to "
            f"type {o_val!r}. Use an explicit value type (e.g. ``Dict[str, str]`` when values are "
            f"strings)."
        )
    if py_val != o_val:
        if py_val == "integer" and o_val == "number":
            return
        raise AssertionError(
            f"{contract_name} ({side}): dict value type mismatch in {model_name}{path_msg}: "
            f"OpenAPI ``additionalProperties`` implies {o_val!r}; Python field {field_name!r} "
            f"value type implies {py_val!r}."
        )


def _assert_python_dict_key_matches_openapi_field(
    parent_model: Type[BaseModel],
    field_name: str,
    openapi_prop_schema: Dict[str, Any],
    *,
    contract_name: str,
    side: str,
    model_name: str,
    openapi_schemas: Dict[str, Dict[str, Any]],
    defs: Dict[str, Dict[str, Any]],
    property_path: str,
) -> None:
    """Assert a model field's ``Dict`` typing matches OpenAPI for that property.

    Args:
        parent_model (Type[BaseModel]): Model class declaring ``field_name``.
        field_name (str): Field on ``parent_model``.
        openapi_prop_schema (Dict[str, Any]): OpenAPI JSON Schema for that property.
        contract_name (str): Contract label for errors.
        side (str): ``request`` or ``response``.
        model_name (str): Model name for errors.
        openapi_schemas (Dict[str, Dict[str, Any]]): OpenAPI ``components/schemas``.
        defs (Dict[str, Dict[str, Any]]): Pydantic ``$defs``.
        property_path (str): Dotted path for error messages.
    """
    field = parent_model.model_fields.get(field_name)
    if field is None:
        return
    resolved = _get_resolved_prop_schema(openapi_prop_schema, openapi_schemas=openapi_schemas)
    if not isinstance(resolved, dict) or resolved.get("type") != "object":
        return
    path_msg = f" for property {property_path!r}" if property_path else ""
    _assert_python_dict_annotation_vs_openapi_object(
        field.annotation,
        resolved,
        contract_name=contract_name,
        side=side,
        model_name=model_name,
        path_msg=path_msg,
        field_name=field_name,
        openapi_schemas=openapi_schemas,
        defs=defs,
    )


def _assert_python_list_element_dict_vs_openapi_array(
    parent_model: Type[BaseModel],
    field_name: str,
    openapi_array_resolved: Dict[str, Any],
    *,
    contract_name: str,
    side: str,
    model_name: str,
    openapi_schemas: Dict[str, Dict[str, Any]],
    defs: Dict[str, Dict[str, Any]],
    property_path: str,
) -> None:
    """Apply dict key/value rules for ``List[Dict[...]]`` vs array-of-object OpenAPI.

    Args:
        parent_model (Type[BaseModel]): Model class declaring ``field_name``.
        field_name (str): Field annotated with ``List[Dict[...]]``.
        openapi_array_resolved (Dict[str, Any]): Resolved OpenAPI schema (``type: array``).
        contract_name (str): Contract label for errors.
        side (str): ``request`` or ``response``.
        model_name (str): Model name for errors.
        openapi_schemas (Dict[str, Dict[str, Any]]): OpenAPI ``components/schemas``.
        defs (Dict[str, Dict[str, Any]]): Pydantic ``$defs``.
        property_path (str): Path to the array field (before ``[*]``).
    """
    field = parent_model.model_fields.get(field_name)
    if field is None:
        return
    if openapi_array_resolved.get("type") != "array":
        return
    ann = _unwrap_annotation(field.annotation)
    origin = get_origin(ann)
    args = get_args(ann)
    if origin not in (list, List) or not args:
        return
    elem_ann = _unwrap_annotation(args[0])
    if get_origin(elem_ann) not in (dict, Dict):
        return
    o_items = openapi_array_resolved.get("items")
    if not isinstance(o_items, dict):
        return
    obj_resolved = _get_resolved_prop_schema(o_items, openapi_schemas=openapi_schemas)
    if not isinstance(obj_resolved, dict) or obj_resolved.get("type") != "object":
        return
    sub = f"{property_path}[*]" if property_path else "[*]"
    path_msg = f" for property {sub!r}"
    _assert_python_dict_annotation_vs_openapi_object(
        elem_ann,
        obj_resolved,
        contract_name=contract_name,
        side=side,
        model_name=model_name,
        path_msg=path_msg,
        field_name=field_name,
        openapi_schemas=openapi_schemas,
        defs=defs,
    )


def _nested_model_cls(
    model_cls: Optional[Type[BaseModel]], field_name: str
) -> Optional[Type[BaseModel]]:
    if model_cls is None:
        return None
    field = model_cls.model_fields.get(field_name)
    if field is None:
        return None
    ann = _unwrap_annotation(field.annotation)
    if isinstance(ann, type):
        try:
            if issubclass(ann, BaseModel):
                return ann
        except TypeError:
            return None
    return None


def _list_item_model_cls(
    model_cls: Optional[Type[BaseModel]], field_name: str
) -> Optional[Type[BaseModel]]:
    if model_cls is None:
        return None
    field = model_cls.model_fields.get(field_name)
    if field is None:
        return None
    ann = _unwrap_annotation(field.annotation)
    origin = get_origin(ann)
    args = get_args(ann)
    if origin not in (list, List):
        return None
    if not args:
        return None
    elem = _unwrap_annotation(args[0])
    if isinstance(elem, type):
        try:
            if issubclass(elem, BaseModel):
                return elem
        except TypeError:
            return None
    return None


def _schema_props(schema: Dict[str, Any]) -> Set[str]:
    return set((schema.get("properties") or {}).keys())


def _schema_required(schema: Dict[str, Any]) -> Set[str]:
    return set(schema.get("required") or [])


def _assert_object_property_sets_aligned(
    openapi_object: Dict[str, Any],
    model_object: Dict[str, Any],
    *,
    contract_name: str,
    side: str,
    model_name: str,
    property_path: str,
) -> None:
    """Assert OpenAPI and SDK agree on property names and ``required`` at one object level.

    Used at the contract body root and, when gated, for nested objects that expose
    ``properties`` on both sides (fixed-shape objects, not map-only ``Dict`` schemas).

    Args:
        openapi_object (Dict[str, Any]): OpenAPI JSON Schema object (this level).
        model_object (Dict[str, Any]): SDK / Pydantic JSON Schema object (this level).
        contract_name (str): Contract label for errors.
        side (str): ``request`` or ``response``.
        model_name (str): Pydantic model name for errors.
        property_path (str): Dotted path for nested errors (may be empty at root).

    Raises:
        AssertionError: If required or property-name sets disagree.
    """
    openapi_required = _schema_required(openapi_object)
    model_required = _schema_required(model_object)
    openapi_names = _schema_props(openapi_object)
    model_names = _schema_props(model_object)
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


def _unwrap_optional(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return the non-null branch of ``anyOf: [T, null]``, else ``schema``.

    Args:
        schema (Dict[str, Any]): A JSON Schema dict.

    Returns:
        Dict[str, Any]: Unwrapped schema dict.
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
    """Resolve ``$ref`` against OpenAPI or Pydantic schema registries.

    Args:
        ref (str): JSON Schema ``$ref`` string.
        openapi_schemas (Optional[Dict[str, Dict[str, Any]]]): OpenAPI schemas map, if any.
        defs (Optional[Dict[str, Dict[str, Any]]]): Pydantic ``$defs`` map, if any.

    Returns:
        Optional[Dict[str, Any]]: Resolved schema dict, or None if unknown.
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
    """Return JSON Schema ``type`` after optional unwrap and shallow ``$ref``.

    Args:
        prop_schema (Dict[str, Any]): Property schema fragment.
        openapi_schemas (Optional[Dict[str, Dict[str, Any]]]): For OpenAPI ``$ref`` resolution.
        defs (Optional[Dict[str, Dict[str, Any]]]): For Pydantic ``$ref`` resolution.

    Returns:
        Optional[str]: Type string (e.g. ``string``, ``object``), or None.
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
    """Return a concrete schema dict (unwrap optional, one-hop ``$ref``).

    Args:
        prop_schema (Dict[str, Any]): Property schema fragment.
        openapi_schemas (Optional[Dict[str, Dict[str, Any]]]): For OpenAPI ``$ref`` resolution.
        defs (Optional[Dict[str, Dict[str, Any]]]): For Pydantic ``$ref`` resolution.

    Returns:
        Optional[Dict[str, Any]]: Resolved dict, or None.
    """
    if not isinstance(prop_schema, dict):
        return None
    schema = _unwrap_optional(prop_schema)
    ref = schema.get("$ref")
    if isinstance(ref, str):
        return _resolve_ref(ref, openapi_schemas, defs)
    return schema


def _infer_homogeneous_property_value_json_type(
    resolved_object: Dict[str, Any],
    *,
    openapi_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
    defs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[str]:
    """Return the common JSON type when every ``properties`` value shares one type.

    Args:
        resolved_object (Dict[str, Any]): OpenAPI/SDK object schema with ``properties``.
        openapi_schemas (Optional[Dict[str, Dict[str, Any]]]): For OpenAPI ``$ref`` resolution.
        defs (Optional[Dict[str, Dict[str, Any]]]): For Pydantic ``$ref`` resolution.

    Returns:
        Optional[str]: Common value type, or None if not a single uniform type.
    """
    if resolved_object.get("type") != "object":
        return None
    ap = resolved_object.get("additionalProperties")
    if ap is True:
        return None
    props = resolved_object.get("properties") or {}
    if not props:
        return None
    if isinstance(ap, dict):
        return None
    seen: Optional[str] = None
    for pschema in props.values():
        if not isinstance(pschema, dict):
            return None
        t = _effective_json_type(pschema, openapi_schemas=openapi_schemas, defs=defs)
        if t is None:
            return None
        if seen is None:
            seen = t
        elif t != seen:
            return None
    return seen


def _assert_homogeneous_object_vs_dict_ap(
    fixed: Dict[str, Any],
    partner: Dict[str, Any],
    *,
    fixed_is_openapi: bool,
    contract_name: str,
    side: str,
    model_name: str,
    path_msg: str,
    openapi_schemas: Dict[str, Dict[str, Any]],
    defs: Dict[str, Dict[str, Any]],
) -> None:
    """Compare homogeneous fixed-key object to a partner ``Dict`` schema.

    Args:
        fixed (Dict[str, Any]): Object whose ``properties`` values share one type.
        partner (Dict[str, Any]): Object with only ``additionalProperties`` typing.
        fixed_is_openapi (bool): True if ``fixed`` is the OpenAPI schema.
        contract_name (str): Contract label for errors.
        side (str): ``request`` or ``response``.
        model_name (str): Model name for errors.
        path_msg (str): Path suffix for errors.
        openapi_schemas (Dict[str, Dict[str, Any]]): OpenAPI ``components/schemas``.
        defs (Dict[str, Dict[str, Any]]): Pydantic ``$defs``.

    Raises:
        AssertionError: If uniform value type disagrees with map value type.
    """
    ap = partner.get("additionalProperties")
    if not isinstance(ap, dict) or (partner.get("properties") or {}):
        return
    if fixed_is_openapi:
        hom = _infer_homogeneous_property_value_json_type(
            fixed, openapi_schemas=openapi_schemas, defs=None
        )
        val = _effective_json_type(ap, defs=defs)
        if hom is not None and val is not None and val != hom:
            raise AssertionError(
                f"{contract_name} ({side}): dict value type mismatch in {model_name}{path_msg}: "
                f"OpenAPI field types are uniformly {hom!r}, SDK Dict value type is {val!r}."
            )
    else:
        hom = _infer_homogeneous_property_value_json_type(
            fixed, openapi_schemas=openapi_schemas, defs=defs
        )
        val = _effective_json_type(ap, openapi_schemas=openapi_schemas)
        if hom is not None and val is not None and val != hom:
            raise AssertionError(
                f"{contract_name} ({side}): dict value type mismatch in {model_name}{path_msg}: "
                f"OpenAPI Dict value type is {val!r}, SDK field types are uniformly {hom!r}."
            )


def _assert_json_schemas_compatible(
    openapi_prop: Dict[str, Any],
    model_prop: Dict[str, Any],
    *,
    contract_name: str,
    side: str,
    model_name: str,
    openapi_schemas: Dict[str, Dict[str, Any]],
    defs: Dict[str, Dict[str, Any]],
    depth: int,
    property_path: str,
    parent_model_for_field: Optional[Type[BaseModel]] = None,
    field_name: Optional[str] = None,
    model_cls: Optional[Type[BaseModel]] = None,
) -> None:
    """Recursively compare OpenAPI and SDK JSON Schemas for nested containers.

    Args:
        openapi_prop (Dict[str, Any]): OpenAPI schema fragment.
        model_prop (Dict[str, Any]): Pydantic JSON schema fragment.
        contract_name (str): Contract label for errors.
        side (str): ``request`` or ``response``.
        model_name (str): Model name for errors.
        openapi_schemas (Dict[str, Dict[str, Any]]): OpenAPI ``components/schemas``.
        defs (Dict[str, Dict[str, Any]]): Pydantic ``$defs``.
        depth (int): Remaining recursion depth.
        property_path (str): Dotted path for errors (may include ``[*]``).
        parent_model_for_field (Optional[Type[BaseModel]]): Model for Python dict checks.
        field_name (Optional[str]): Field on ``parent_model_for_field`` for dict/list checks.
        model_cls (Optional[Type[BaseModel]]): Current nested Pydantic model for property walks.

    Raises:
        AssertionError: If shapes or types disagree.
    """
    if depth <= 0:
        return
    if parent_model_for_field is not None and field_name:
        ro0 = _get_resolved_prop_schema(openapi_prop, openapi_schemas=openapi_schemas)
        if isinstance(ro0, dict) and ro0.get("type") == "object":
            _assert_python_dict_key_matches_openapi_field(
                parent_model_for_field,
                field_name,
                openapi_prop,
                contract_name=contract_name,
                side=side,
                model_name=model_name,
                openapi_schemas=openapi_schemas,
                defs=defs,
                property_path=property_path,
            )
    o = _get_resolved_prop_schema(openapi_prop, openapi_schemas=openapi_schemas)
    m = _get_resolved_prop_schema(model_prop, defs=defs)
    if not isinstance(o, dict) or not isinstance(m, dict):
        return

    o_type = o.get("type")
    m_type = m.get("type")
    if o_type is None and m_type is None:
        return
    if o_type is None or m_type is None:
        pm = f" for property {property_path!r}" if property_path else ""
        _raise_unresolved_json_type(
            contract_name,
            side,
            f"nested schema{pm} in {model_name}",
            o_type,
            m_type,
            property_path,
        )
    if o_type != m_type:
        path_msg = f" for property {property_path!r}" if property_path else ""
        raise AssertionError(
            f"{contract_name} ({side}): nested schema{path_msg} in {model_name}: "
            f"type mismatch: OpenAPI has {o_type!r}, SDK has {m_type!r}"
        )

    if o_type in _JSON_SCALAR_TYPES:
        return

    if o_type == "array":
        # Tuple / positional array schemas — skip element-type rules
        if "prefixItems" in o or "prefixItems" in m:
            return
        o_items = o.get("items")
        m_items = m.get("items")
        o_items_dict = isinstance(o_items, dict)
        m_items_dict = isinstance(m_items, dict)
        path_msg = f" for property {property_path!r}" if property_path else ""
        if o_items_dict and m_items_dict:
            if parent_model_for_field is not None and field_name:
                _assert_python_list_element_dict_vs_openapi_array(
                    parent_model_for_field,
                    field_name,
                    o,
                    contract_name=contract_name,
                    side=side,
                    model_name=model_name,
                    openapi_schemas=openapi_schemas,
                    defs=defs,
                    property_path=property_path,
                )
            sub = f"{property_path}[*]" if property_path else "[*]"
            assert isinstance(o_items, dict) and isinstance(m_items, dict)
            _assert_json_schemas_compatible(
                o_items,
                m_items,
                contract_name=contract_name,
                side=side,
                model_name=model_name,
                openapi_schemas=openapi_schemas,
                defs=defs,
                depth=depth - 1,
                property_path=sub,
                parent_model_for_field=None,
                field_name=None,
                model_cls=model_cls,
            )
        elif o_items_dict ^ m_items_dict:
            raise AssertionError(
                f"{contract_name} ({side}): array element type mismatch in {model_name}{path_msg}: "
                "OpenAPI and SDK must both use a schema object for ``items`` (or neither). "
                f"OpenAPI has typed items: {o_items_dict}, SDK has typed items: {m_items_dict}."
            )
        return

    if o_type == "object":
        o_ap = o.get("additionalProperties")
        m_ap = m.get("additionalProperties")
        o_props = o.get("properties") or {}
        m_props = m.get("properties") or {}
        path_msg = f" for property {property_path!r}" if property_path else ""

        if isinstance(o_ap, dict) and isinstance(m_ap, dict):
            sub = f"{property_path}.__value__" if property_path else "__value__"
            _assert_json_schemas_compatible(
                o_ap,
                m_ap,
                contract_name=contract_name,
                side=side,
                model_name=model_name,
                openapi_schemas=openapi_schemas,
                defs=defs,
                depth=depth - 1,
                property_path=sub,
                parent_model_for_field=None,
                field_name=None,
                model_cls=None,
            )

        # OpenAPI ``additionalProperties: true`` means any value type; SDK cannot narrow to
        # e.g. ``Dict[str, int]`` without updating the spec. SDK ``Dict[str, Any]`` (true) with
        # OpenAPI typed values is allowed — the client may be looser than the documented API.
        if o_ap is True and isinstance(m_ap, dict):
            raise AssertionError(
                f"{contract_name} ({side}): OpenAPI allows any dict values "
                f"(additionalProperties: true) in {model_name}{path_msg} but the SDK schema "
                f"uses a typed additionalProperties value schema. Use Dict[str, Any] or match "
                f"OpenAPI."
            )

        _assert_homogeneous_object_vs_dict_ap(
            o,
            m,
            fixed_is_openapi=True,
            contract_name=contract_name,
            side=side,
            model_name=model_name,
            path_msg=path_msg,
            openapi_schemas=openapi_schemas,
            defs=defs,
        )
        _assert_homogeneous_object_vs_dict_ap(
            m,
            o,
            fixed_is_openapi=False,
            contract_name=contract_name,
            side=side,
            model_name=model_name,
            path_msg=path_msg,
            openapi_schemas=openapi_schemas,
            defs=defs,
        )

        # Fixed-shape nested objects: full required / missing / extra parity when both sides
        # list ``properties``. If either side is map-only (empty ``properties``), skip so
        # ``Dict[str, …]`` vs explicit OpenAPI properties still works.
        if o_props and m_props:
            _assert_object_property_sets_aligned(
                o,
                m,
                contract_name=contract_name,
                side=side,
                model_name=model_name,
                property_path=property_path,
            )

        # Nested models (e.g. ``constraints.immutable_effect``): compare only names that
        # exist on both sides so ``Dict[str, …]`` vs explicit OpenAPI properties still works.
        for prop in sorted(set(o_props) & set(m_props)):
            sub_path = f"{property_path}.{prop}" if property_path else prop
            if model_cls is not None:
                _assert_python_dict_key_matches_openapi_field(
                    model_cls,
                    prop,
                    o_props[prop],
                    contract_name=contract_name,
                    side=side,
                    model_name=model_name,
                    openapi_schemas=openapi_schemas,
                    defs=defs,
                    property_path=sub_path,
                )
            ot = _effective_json_type(o_props[prop], openapi_schemas=openapi_schemas)
            mt = _effective_json_type(m_props[prop], defs=defs)
            if ot is None and mt is None:
                continue
            if ot is None or mt is None:
                _raise_unresolved_json_type(
                    contract_name,
                    side,
                    f"property {prop!r} in {model_name} for property {sub_path!r}",
                    ot,
                    mt,
                    sub_path,
                )
            if ot != mt:
                raise AssertionError(
                    f"{contract_name} ({side}): property {prop!r} type mismatch in {model_name} "
                    f"for property {sub_path!r}: OpenAPI has {ot!r}, SDK has {mt!r}"
                )
            if ot == "object" and mt == "object":
                _assert_json_schemas_compatible(
                    o_props[prop],
                    m_props[prop],
                    contract_name=contract_name,
                    side=side,
                    model_name=model_name,
                    openapi_schemas=openapi_schemas,
                    defs=defs,
                    depth=depth - 1,
                    property_path=sub_path,
                    parent_model_for_field=None,
                    field_name=None,
                    model_cls=_nested_model_cls(model_cls, prop),
                )
            elif ot == "array" and mt == "array":
                _assert_json_schemas_compatible(
                    o_props[prop],
                    m_props[prop],
                    contract_name=contract_name,
                    side=side,
                    model_name=model_name,
                    openapi_schemas=openapi_schemas,
                    defs=defs,
                    depth=depth - 1,
                    property_path=sub_path,
                    parent_model_for_field=model_cls,
                    field_name=prop,
                    model_cls=_list_item_model_cls(model_cls, prop),
                )


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
    model_cls: Optional[Type[BaseModel]] = None,
) -> None:
    """Compare top-level OpenAPI and SDK object schemas and recurse into fields.

    Args:
        openapi_schema (Dict[str, Any]): OpenAPI object schema (body or nested).
        model_schema (Dict[str, Any]): Pydantic ``model_json_schema()`` object fragment.
        contract_name (str): Contract label for errors.
        side (str): ``request`` or ``response``.
        model_name (str): Model name for errors.
        openapi_schemas (Dict[str, Dict[str, Any]]): OpenAPI ``components/schemas``.
        root_model_defs (Optional[Dict[str, Dict[str, Any]]]): Root SDK ``$defs``.
        depth (int): Max recursion depth.
        property_path (str): Prefix for nested property paths in errors.
        model_cls (Optional[Type[BaseModel]]): Pydantic model for Python dict annotations.

    Raises:
        AssertionError: If required fields, names, or types disagree.
    """
    if depth <= 0:
        return
    openapi_props = openapi_schema.get("properties") or {}
    model_props = model_schema.get("properties") or {}
    defs = root_model_defs if root_model_defs is not None else (model_schema.get("$defs") or {})
    _assert_object_property_sets_aligned(
        openapi_schema,
        model_schema,
        contract_name=contract_name,
        side=side,
        model_name=model_name,
        property_path=property_path,
    )
    openapi_names = _schema_props(openapi_schema)
    model_names = _schema_props(model_schema)

    for prop in openapi_names & model_names:
        sub_path = f"{property_path}.{prop}" if property_path else prop
        openapi_type = _effective_json_type(openapi_props[prop], openapi_schemas=openapi_schemas)
        model_type = _effective_json_type(model_props[prop], defs=defs)
        if openapi_type is None and model_type is None:
            continue
        if openapi_type is None or model_type is None:
            _raise_unresolved_json_type(
                contract_name,
                side,
                f"property {prop!r} in {model_name} at {sub_path!r}",
                openapi_type,
                model_type,
                sub_path,
            )
        if openapi_type != model_type:
            raise AssertionError(
                f"{contract_name} ({side}): property {prop!r} type mismatch in "
                f"{model_name}: OpenAPI has {openapi_type!r}, SDK has {model_type!r}"
            )
        if openapi_type == "object" and model_type == "object":
            _assert_json_schemas_compatible(
                openapi_props[prop],
                model_props[prop],
                contract_name=contract_name,
                side=side,
                model_name=model_name,
                openapi_schemas=openapi_schemas,
                defs=defs,
                depth=depth - 1,
                property_path=sub_path,
                parent_model_for_field=model_cls,
                field_name=prop,
                model_cls=_nested_model_cls(model_cls, prop),
            )
        elif openapi_type == "array" and model_type == "array":
            _assert_json_schemas_compatible(
                openapi_props[prop],
                model_props[prop],
                contract_name=contract_name,
                side=side,
                model_name=model_name,
                openapi_schemas=openapi_schemas,
                defs=defs,
                depth=depth - 1,
                property_path=sub_path,
                parent_model_for_field=model_cls,
                field_name=prop,
                model_cls=_list_item_model_cls(model_cls, prop),
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
                model_cls=contract.request_model,
            )
