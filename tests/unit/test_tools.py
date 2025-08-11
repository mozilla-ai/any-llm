import pytest

from any_llm.tools import callable_to_tool, prepare_tools


def test_callable_to_tool_basic() -> None:
    """Test basic callable to tool conversion."""

    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    tool = callable_to_tool(add_numbers)

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "add_numbers"
    assert tool["function"]["description"] == "Add two numbers together."

    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"a", "b"}
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "integer"


def test_callable_to_tool_with_optional_params() -> None:
    """Test callable with optional parameters."""

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    tool = callable_to_tool(greet)

    params = tool["function"]["parameters"]
    assert params["required"] == ["name"]  # Only required parameter
    assert "greeting" in params["properties"]
    assert params["properties"]["name"]["type"] == "string"
    assert params["properties"]["greeting"]["type"] == "string"


def test_callable_to_tool_missing_docstring() -> None:
    """Test that function without docstring raises error."""

    def no_doc_function(x: int) -> int:
        return x

    with pytest.raises(ValueError, match="must have a docstring"):
        callable_to_tool(no_doc_function)


def test_prepare_tools_mixed() -> None:
    """Test prepare_tools with mix of callables and dicts."""

    def my_function(x: int) -> int:
        """My function."""
        return x

    existing_tool = {
        "type": "function",
        "function": {
            "name": "existing_tool",
            "description": "An existing tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    tools = prepare_tools([my_function, existing_tool])

    assert len(tools) == 2
    assert tools[0]["function"]["name"] == "my_function"
    assert tools[1]["function"]["name"] == "existing_tool"
