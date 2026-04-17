from any_llm.providers.llama.utils import _patch_json_schema


def test_patch_json_schema_none_parameters() -> None:
    """Regression: parameters=None must not raise 'NoneType' object is not subscriptable."""
    schema: dict = {"type": "function", "function": {"name": "ping", "parameters": None}}
    result = _patch_json_schema(schema)
    assert result is schema


def test_patch_json_schema_parameters_missing_properties() -> None:
    """Regression: parameters without 'properties' must not raise KeyError."""
    schema: dict = {"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}
    result = _patch_json_schema(schema)
    assert result is schema


def test_patch_json_schema_patches_union_without_type() -> None:
    """Existing behaviour: oneOf props without type get type=string."""
    schema: dict = {
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
