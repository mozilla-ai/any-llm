import json
from typing import Any, Literal


def _patch_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Patch the JSON schema to be compatible with Llama's API."""
    # Llama requires that the 'union_specified' has a parameter type specified
    # so we need to patch the schema to include the type of the parameter
    # if any of the function call parameter properties have 'oneOf' set, make sure the property has type set. If not, set it to string. This is a quirk with Llama API currently.
    props = schema["function"]["parameters"]["properties"]
    for prop in props:
        if "oneOf" in props[prop] and "type" not in props[prop]:
            props[prop]["type"] = "string"

    return schema


def _map_stop_reason_to_finish_reason(
    stop_reason: str | None,
) -> Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None:
    """Map Llama stop reasons to OpenAI finish reasons."""
    if not stop_reason:
        return None

    mapping: dict[str, Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = {
        "end_of_turn": "stop",
        "tool_call": "tool_calls",
        "max_tokens": "length",
    }
    return mapping.get(stop_reason, "stop")


def _extract_content_text(content: Any) -> str | None:
    """Extract text from Llama content which may be a string, object, or dict."""
    if content is None:
        return None
    if isinstance(content, str):
        return content

    # Try to get text from content object (e.g., TextContent)
    text_content = getattr(content, "text", None)
    if text_content is not None:
        return text_content

    if isinstance(content, dict):
        return content.get("text", str(content))

    return str(content) if content else None


def _convert_tool_call_arguments(arguments: Any) -> str:
    """Convert tool call arguments to JSON string format."""
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        return json.dumps(arguments)
    return str(arguments)
