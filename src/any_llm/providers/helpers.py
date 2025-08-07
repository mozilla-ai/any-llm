from typing import Any, Optional, Sequence
import json

from any_llm.types.completion import (
    ChatCompletion,
    Choice,
    CompletionUsage,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
    Function,
    Reasoning,
)


def create_tool_calls_from_list(
    tool_calls_data: Sequence[dict[str, Any]],
) -> list[ChatCompletionMessageToolCall]:
    """
    Convert a list of tool call dictionaries to ChatCompletionMessageFunctionToolCall objects.

    Handles common variations in tool call structure across providers.
    """
    tool_calls: list[ChatCompletionMessageToolCall] = []

    for raw_tool_call in tool_calls_data:
        tool_call = dict(raw_tool_call)
        # Extract tool call ID (handle various formats)
        tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id") or f"call_{hash(str(tool_call))}"

        function_info = tool_call.get("function", {})
        if not function_info and "name" in tool_call:
            function_info = {
                "name": tool_call["name"],
                "arguments": tool_call.get("arguments", tool_call.get("input", {})),
            }

        name_value = function_info.get("name")
        if not isinstance(name_value, str) or not name_value:
            # Skip invalid tool calls without a name
            continue

        arguments_value = function_info.get("arguments")
        arguments: str

        # Ensure arguments is a JSON string when appropriate
        if isinstance(arguments_value, (dict, list)):
            arguments = json.dumps(arguments_value)
        elif isinstance(arguments_value, str):
            arguments = arguments_value
        elif arguments_value is None:
            arguments = "{}"
        else:
            arguments = str(arguments_value)

        tool_calls.append(
            ChatCompletionMessageFunctionToolCall(
                id=tool_call_id,
                type="function",
                function=Function(name=name_value, arguments=arguments),
            )
        )

    return tool_calls


def create_choice_from_message_data(
    message_data: dict[str, Any],
    index: int = 0,
    finish_reason: str = "stop",
    finish_reason_mapping: Optional[dict[str, str]] = None,
) -> Choice:
    """
    Create a Choice object from message data, handling tool calls and content.

    Args:
        message_data: Dictionary containing message content and tool calls
        index: Choice index (default 0)
        finish_reason: Raw finish reason from provider
        finish_reason_mapping: Optional mapping to convert provider finish reasons to OpenAI format
    """
    # Normalize finish reason
    allowed_finish_reasons: set[str] = {"stop", "length", "tool_calls", "content_filter", "function_call"}
    default_finish_reason_mapping = {"max_tokens": "length", "tool_use": "tool_calls"}
    mapping = finish_reason_mapping or default_finish_reason_mapping
    if finish_reason in mapping:
        finish_reason = mapping[finish_reason]
    if finish_reason not in allowed_finish_reasons:
        finish_reason = "stop"

    # Extract tool calls if present
    raw_tool_calls = message_data.get("tool_calls")
    tool_calls = create_tool_calls_from_list(raw_tool_calls) if raw_tool_calls else None

    # Create the message
    message = ChatCompletionMessage(
        role="assistant",
        content=message_data.get("content"),
        tool_calls=tool_calls,
        reasoning=Reasoning(content=str(message_data.get("reasoning_content")))
        if message_data.get("reasoning_content")
        else None,
    )

    return Choice(
        finish_reason=finish_reason,  # type: ignore[arg-type]
        index=index,
        message=message,
    )


def create_usage_from_data(
    usage_data: dict[str, Any], token_field_mapping: Optional[dict[str, str]] = None
) -> CompletionUsage:
    """
    Create CompletionUsage from provider usage data.

    Args:
        usage_data: Dictionary containing usage information
        token_field_mapping: Optional mapping for field names (e.g., {"input_tokens": "prompt_tokens"})
    """
    default_mapping = {
        "completion_tokens": "completion_tokens",
        "prompt_tokens": "prompt_tokens",
        "total_tokens": "total_tokens",
    }

    if token_field_mapping:
        for openai_field, provider_field in token_field_mapping.items():
            if provider_field in usage_data:
                default_mapping[openai_field] = provider_field

    return CompletionUsage(
        completion_tokens=usage_data.get(default_mapping["completion_tokens"], 0),
        prompt_tokens=usage_data.get(default_mapping["prompt_tokens"], 0),
        total_tokens=usage_data.get(default_mapping["total_tokens"], 0),
    )


def create_completion_from_response(
    response_data: dict[str, Any],
    model: str,
    provider_name: str = "provider",
    finish_reason_mapping: Optional[dict[str, str]] = None,
    token_field_mapping: Optional[dict[str, str]] = None,
    id_field: str = "id",
    created_field: str = "created",
    choices_field: str = "choices",
    usage_field: str = "usage",
) -> ChatCompletion:
    """
    Create a complete ChatCompletion from provider response data.

    This is the main utility that most providers can use to convert their responses.

    Args:
        response_data: The raw response from the provider
        model: Model name to use in the response
        provider_name: Name of the provider (for generating fallback IDs)
        finish_reason_mapping: Mapping for finish reasons
        token_field_mapping: Mapping for token count fields
        id_field: Field name for response ID
        created_field: Field name for creation timestamp
        choices_field: Field name for choices array
        usage_field: Field name for usage data
    """
    choices: list[Choice] = []
    choices_data = response_data.get(choices_field, [])

    if not choices_data and "message" in response_data:
        choices_data = [
            {"message": response_data["message"], "finish_reason": response_data.get("finish_reason", "stop")}
        ]

    for i, choice_data in enumerate(choices_data):
        choice = create_choice_from_message_data(
            choice_data.get("message", choice_data),
            index=i,
            finish_reason=choice_data.get("finish_reason", "stop"),
            finish_reason_mapping=finish_reason_mapping,
        )
        choices.append(choice)
    usage = None
    if usage_field in response_data and response_data[usage_field]:
        usage = create_usage_from_data(response_data[usage_field], token_field_mapping)

    response_id = response_data.get(id_field, f"{provider_name}_{hash(str(response_data))}")

    return ChatCompletion(
        id=response_id,
        model=model,
        choices=choices,
        usage=usage,
        created=response_data.get(created_field, 0),
        object="chat.completion",
    )
