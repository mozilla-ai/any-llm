"""Tools utilities for converting Python callables to OpenAI tools format."""

import inspect
from typing import Any, Callable, Union, get_type_hints


def callable_to_tool(func: Callable[..., Any]) -> dict[str, Any]:
    """Convert a Python callable to OpenAI tools format.

    Args:
        func: A Python callable (function) to convert to a tool

    Returns:
        Dictionary in OpenAI tools format

    Raises:
        ValueError: If the function doesn't have proper docstring or type annotations

    Example:
        >>> def get_weather(location: str, unit: str = "celsius") -> str:
        ...     '''Get weather information for a location.'''
        ...     return f"Weather in {location} is sunny, 25°{unit[0].upper()}"
        >>>
        >>> tool = callable_to_tool(get_weather)
        >>> # Returns OpenAI tools format dict
    """
    # Validate the function has a docstring
    if not func.__doc__:
        raise ValueError(f"Function {func.__name__} must have a docstring")

    sig = inspect.signature(func)

    type_hints = get_type_hints(func)

    # Build parameter schema
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        param_type = type_hints.get(param_name, str)

        json_type = _python_type_to_json_schema_type(param_type)

        properties[param_name] = {
            "type": json_type,
            "description": f"Parameter {param_name} of type {param_type.__name__}",
        }

        # Add to required if no default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    tool_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__.strip(),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }

    return tool_schema


def _python_type_to_json_schema_type(python_type: type) -> str:
    """Convert Python type to JSON Schema type string."""
    if python_type is str:
        return "string"
    elif python_type is int:
        return "integer"
    elif python_type is float:
        return "number"
    elif python_type is bool:
        return "boolean"
    elif python_type is list:
        return "array"
    elif python_type is dict:
        return "object"
    else:
        # Default to string for unknown types
        return "string"


def prepare_tools(tools: list[Union[dict[str, Any], Callable[..., Any]]]) -> list[dict[str, Any]]:
    """Prepare tools for completion API by converting callables to OpenAI format.

    Args:
        tools: List of tools, can be mix of callables and already formatted tool dicts

    Returns:
        List of tools in OpenAI format

    Example:
        >>> def add(a: int, b: int) -> int:
        ...     '''Add two numbers.'''
        ...     return a + b
        >>>
        >>> tools = prepare_tools([add])
        >>> # Returns list of OpenAI format tool dicts
    """
    prepared_tools = []

    for tool in tools:
        if callable(tool):
            prepared_tools.append(callable_to_tool(tool))
        elif isinstance(tool, dict):
            # Already in tool format
            prepared_tools.append(tool)
        else:
            raise ValueError(f"Tool must be callable or dict, got {type(tool)}")

    return prepared_tools
