from any_llm.providers.google.utils import _convert_tool_spec


def test_convert_tool_spec_basic_mapping() -> None:
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search things",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query"},
                        # Array without items → should default items to {"type": "string"}
                        "opts": {"type": "array"},
                        # Array with items → should be preserved
                        "count_list": {"type": "array", "items": {"type": "integer"}},
                        # Enum should be preserved
                        "mode": {"type": "string", "enum": ["a", "b"]},
                        # additionalProperties should be dropped
                        "config": {"type": "object", "additionalProperties": {"type": "integer"}},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    tools = _convert_tool_spec(openai_tools)

    assert len(tools) == 1
    assert tools[0].function_declarations[0].name == "search"  # type: ignore[index]
    assert tools[0].function_declarations[0].description == "Search things"  # type: ignore[index]
    assert tools[0].function_declarations[0].parameters.type == "OBJECT"  # type: ignore[index, union-attr]
    assert tools[0].function_declarations[0].parameters.properties["query"].type == "STRING"  # type: ignore[union-attr, index]
    assert tools[0].function_declarations[0].parameters.properties["opts"].type == "ARRAY"  # type: ignore[union-attr, index]
    assert tools[0].function_declarations[0].parameters.properties["count_list"].type == "ARRAY"  # type: ignore[union-attr, index]
    assert tools[0].function_declarations[0].parameters.properties["mode"].type == "STRING"  # type: ignore[union-attr, index]
    assert tools[0].function_declarations[0].parameters.properties["config"].type == "OBJECT"  # type: ignore[union-attr, index]
    assert "additionalProperties" not in tools[0].function_declarations[0].parameters.properties["config"]  # type: ignore[union-attr, index]
