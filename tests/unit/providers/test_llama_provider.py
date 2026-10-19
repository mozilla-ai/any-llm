from typing import Any

from any_llm.providers.llama.utils import _patch_json_schema


def test_patch_json_schema_none_parameters() -> None:
    """Regression: parameters=None must not raise 'NoneType' object is not subscriptable."""
    schema: dict[str, Any] = {"type": "function", "function": {"name": "ping", "parameters": None}}
    result = _patch_json_schema(schema)
    assert result is schema


def test_patch_json_schema_parameters_missing_properties() -> None:
    """Regression: parameters without 'properties' must not raise KeyError."""
    schema: dict[str, Any] = {"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}
    result = _patch_json_schema(schema)
    assert result is schema


def test_patch_json_schema_patches_union_without_type() -> None:
    """Existing behaviour: oneOf props without type get type=string."""
    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "fn",
            "parameters": {
                "type": "object",
                "properties": {"x": {"oneOf": [{"type": "string"}]}},
            },
        },
    }
    _patch_json_schema(schema)
    assert schema["function"]["parameters"]["properties"]["x"]["type"] == "string"


def test_patch_json_schema_patches_any_of_union_without_type() -> None:
    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "fn",
            "parameters": {
                "type": "object",
                "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
            },
        },
    }
    _patch_json_schema(schema)
    assert schema["function"]["parameters"]["properties"]["x"]["type"] == "string"


def test_patch_json_schema_leaves_typed_and_non_union_props_unchanged() -> None:
    typed_union = {"anyOf": [{"type": "integer"}, {"type": "number"}], "type": "number"}
    plain = {"type": "integer"}
    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "fn",
            "parameters": {
                "type": "object",
                "properties": {"typed_union": dict(typed_union), "plain": dict(plain)},
            },
        },
    }
    _patch_json_schema(schema)
    props = schema["function"]["parameters"]["properties"]
    assert props["typed_union"] == typed_union
    assert props["plain"] == plain
