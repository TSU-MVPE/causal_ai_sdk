"""Unit tests for dataset_schema.validate_columns_data."""

from __future__ import annotations

import pytest
from causal_ai_sdk.exceptions import ValidationError
from causal_ai_sdk.utils.dataset_schema import validate_columns_data


def test_validate_columns_data_valid_returns_columns_and_row_count():
    """Valid dict with columns and data returns (columns, row_count)."""
    obj = {"columns": ["A", "B"], "data": [[1, 2], [3, 4]]}
    columns, row_count = validate_columns_data(obj)
    assert columns == ["A", "B"]
    assert row_count == 2


def test_validate_columns_data_not_dict_raises():
    """Non-dict input raises ValidationError with context in message."""
    with pytest.raises(ValidationError) as exc_info:
        validate_columns_data([], context="Test")
    assert "Test" in exc_info.value.message
    assert "expected object" in exc_info.value.message

    with pytest.raises(ValidationError) as exc_info:
        validate_columns_data("foo", context="Other")
    assert "Other" in exc_info.value.message
    assert "str" in exc_info.value.message


def test_validate_columns_data_missing_keys_raises():
    """Dict missing 'columns' or 'data' raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        validate_columns_data({"data": []}, context="Context")
    assert "Context" in exc_info.value.message
    assert "must contain" in exc_info.value.message

    with pytest.raises(ValidationError):
        validate_columns_data({"columns": []}, context="Context")

    with pytest.raises(ValidationError):
        validate_columns_data({}, context="Context")


def test_validate_columns_data_columns_not_list_raises():
    """'columns' that is not a list raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        validate_columns_data(
            {"columns": "A,B", "data": [[1, 2]]},
            context="Context",
        )
    assert "Context" in exc_info.value.message
    assert "columns" in exc_info.value.message
    assert "list" in exc_info.value.message


def test_validate_columns_data_columns_not_all_strings_raises():
    """'columns' with non-string element raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        validate_columns_data(
            {"columns": ["A", 42], "data": [[1, 2]]},
            context="Context",
        )
    assert "Context" in exc_info.value.message
    assert "strings" in exc_info.value.message
    assert "index 1" in exc_info.value.message


def test_validate_columns_data_data_not_list_raises():
    """'data' that is not a list raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        validate_columns_data(
            {"columns": ["A"], "data": {"row": [1]}},
            context="Context",
        )
    assert "Context" in exc_info.value.message
    assert "data" in exc_info.value.message
    assert "list" in exc_info.value.message


def test_validate_columns_data_row_not_list_raises():
    """Row that is not a list raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        validate_columns_data(
            {"columns": ["A", "B"], "data": [[1, 2], "not-a-row"]},
            context="Context",
        )
    assert "Context" in exc_info.value.message
    assert "row" in exc_info.value.message
    assert "row index 1" in exc_info.value.message


def test_validate_columns_data_row_wrong_length_raises():
    """Row with length != len(columns) raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        validate_columns_data(
            {"columns": ["A", "B"], "data": [[1, 2], [3, 4, 5]]},
            context="Context",
        )
    assert "Context" in exc_info.value.message
    assert "row at index 1" in exc_info.value.message
    assert "length 3" in exc_info.value.message
    assert "expected 2" in exc_info.value.message
