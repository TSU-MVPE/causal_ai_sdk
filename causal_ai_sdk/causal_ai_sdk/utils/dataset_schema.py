"""Validation helpers for dataset JSON schema (columns + data)."""

from typing import Any, List, Tuple

from causal_ai_sdk.exceptions import ValidationError


def validate_columns_data(obj: Any, context: str = "JSON") -> Tuple[List[str], int]:
    """Validate a dict with 'columns' and 'data' and return (columns, row_count).

    Ensures columns is a list of strings and data is a list of rows (list of lists)
    with each row length equal to len(columns).

    Args:
        obj (Any): Parsed JSON object (expected dict with 'columns' and 'data').
        context (str): Short description for error messages
            (e.g. "JSON", "JSON dataset at index 0").

    Returns:
        Tuple[List[str], int]: (column names, number of data rows).

    Raises:
        ValidationError: If structure or types are invalid.
    """
    if not isinstance(obj, dict):
        raise ValidationError(
            message=(
                f"{context}: expected object with 'columns' and 'data', "
                f"got {type(obj).__name__}"
            ),
            status_code=None,
            response_data=None,
        )
    if "columns" not in obj or "data" not in obj:
        raise ValidationError(
            message=f"{context}: must contain 'columns' and 'data' keys",
            status_code=None,
            response_data=None,
        )

    columns = obj["columns"]
    data = obj["data"]

    if not isinstance(columns, list):
        raise ValidationError(
            message=(
                f"{context}: 'columns' must be a list of column names, "
                f"got {type(columns).__name__}"
            ),
            status_code=None,
            response_data=None,
        )
    num_cols = len(columns)
    for i, c in enumerate(columns):
        if not isinstance(c, str):
            raise ValidationError(
                message=(
                    f"{context}: 'columns' must be a list of strings, "
                    f"got {type(c).__name__} at index {i}"
                ),
                status_code=None,
                response_data=None,
            )

    if not isinstance(data, list):
        raise ValidationError(
            message=f"{context}: 'data' must be a list of rows, got {type(data).__name__}",
            status_code=None,
            response_data=None,
        )

    for row_idx, row in enumerate(data):
        if not isinstance(row, list):
            raise ValidationError(
                message=(
                    f"{context}: each row in 'data' must be a list, "
                    f"got {type(row).__name__} at row index {row_idx}"
                ),
                status_code=None,
                response_data=None,
            )
        if len(row) != num_cols:
            raise ValidationError(
                message=(
                    f"{context}: row at index {row_idx} has length {len(row)}, "
                    f"expected {num_cols} (number of columns)"
                ),
                status_code=None,
                response_data=None,
            )

    return list(columns), len(data)
