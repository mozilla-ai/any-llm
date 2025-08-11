import json
from typing import Any, Dict, List

from any_llm.logging import logger

try:
    from anthropic.types import Message
    from anthropic.types import (
        ContentBlockStartEvent,
        ContentBlockDeltaEvent,
        ContentBlockStopEvent,
        MessageStopEvent,
    )
except ImportError:
    msg = "anthropic is not installed. Please install it with `pip install any-llm-sdk[anthropic]`"
    raise ImportError(msg)

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
    Choice,
    CompletionUsage,
    Function,
)

DEFAULT_MAX_TOKENS = 4096


def _convert_messages_for_anthropic(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert messages to Anthropic format, extracting system message."""
    system_message = None
    filtered_messages = []

    for message in messages:
        if message["role"] == "system":
            if system_message is None:
                system_message = message["content"]
            else:
                system_message += "\n" + message["content"]
        else:
            filtered_messages.append(message)

    return system_message, filtered_messages


def _create_openai_chunk_from_anthropic_chunk(chunk: Any) -> ChatCompletionChunk:
    """Convert Anthropic streaming chunk to OpenAI ChatCompletionChunk format."""
    chunk_dict = {
        "id": f"chatcmpl-{hash(str(chunk))}",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "claude-3-5-sonnet-20241022",  # Default model
        "choices": [],
        "usage": None,
    }

    delta: Dict[str, Any] = {}
    finish_reason = None

    if isinstance(chunk, ContentBlockStartEvent):
        # Starting a new content block
        if chunk.content_block.type == "text":
            delta = {"content": ""}
        elif chunk.content_block.type == "tool_use":
            # Start of tool call
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": chunk.content_block.id,
                        "type": "function",
                        "function": {"name": chunk.content_block.name, "arguments": ""},
                    }
                ]
            }

    elif isinstance(chunk, ContentBlockDeltaEvent):
        # Delta content
        if chunk.delta.type == "text_delta":
            delta = {"content": chunk.delta.text}
        elif chunk.delta.type == "input_json_delta":
            # Tool call arguments delta
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": chunk.delta.partial_json},
                    }
                ]
            }

    elif isinstance(chunk, ContentBlockStopEvent):
        # End of content block
        if hasattr(chunk, "content_block") and chunk.content_block.type == "tool_use":
            finish_reason = "tool_calls"
        else:
            finish_reason = None

    elif isinstance(chunk, MessageStopEvent):
        # End of message
        finish_reason = "stop"
        if hasattr(chunk, "message") and chunk.message.usage:
            chunk_dict["usage"] = {
                "prompt_tokens": chunk.message.usage.input_tokens,
                "completion_tokens": chunk.message.usage.output_tokens,
                "total_tokens": chunk.message.usage.input_tokens + chunk.message.usage.output_tokens,
            }

    choice = {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
        "logprobs": None,
    }

    chunk_dict["choices"] = [choice]

    return ChatCompletionChunk.model_validate(chunk_dict)


def _convert_response(response: Message) -> ChatCompletion:
    """Convert Anthropic Message to OpenAI ChatCompletion format."""
    finish_reason_raw = response.stop_reason or "end_turn"
    finish_reason_map = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls"}
    finish_reason = finish_reason_map.get(finish_reason_raw, "stop")

    content_parts: list[str] = []
    tool_calls: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] = []

    for content_block in response.content:
        if content_block.type == "text":
            content_parts.append(content_block.text)
        elif content_block.type == "tool_use":
            tool_calls.append(
                ChatCompletionMessageFunctionToolCall(
                    id=content_block.id,
                    type="function",
                    function=Function(
                        name=content_block.name,
                        arguments=json.dumps(content_block.input),
                    ),
                )
            )
        elif content_block.type == "thinking":
            # Provider does not advertise reasoning support; include in content for completeness
            content_parts.append(content_block.thinking)
        else:
            raise ValueError(f"Unsupported content block type: {content_block.type}")

    message = ChatCompletionMessage(role="assistant", content="".join(content_parts), tool_calls=tool_calls or None)

    usage = CompletionUsage(
        completion_tokens=response.usage.output_tokens,
        prompt_tokens=response.usage.input_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )

    from typing import Literal, cast

    choice = Choice(
        index=0,
        finish_reason=cast(
            Literal["stop", "length", "tool_calls", "content_filter", "function_call"], finish_reason or "stop"
        ),
        message=message,
    )

    created_ts = int(response.created_at.timestamp()) if hasattr(response, "created_at") else 0

    return ChatCompletion(
        id=response.id,
        model=response.model,
        created=created_ts,
        object="chat.completion",
        choices=[choice],
        usage=usage,
    )


def _convert_tool_spec(openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI tool specification to Anthropic format."""
    generic_tools = []

    for tool in openai_tools:
        if tool.get("type") != "function":
            continue

        function = tool["function"]
        generic_tool = {
            "name": function["name"],
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
        }
        generic_tools.append(generic_tool)

    anthropic_tools = []
    for tool in generic_tools:
        anthropic_tool = {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": {
                "type": "object",
                "properties": tool["parameters"]["properties"],
                "required": tool["parameters"].get("required", []),
            },
        }
        anthropic_tools.append(anthropic_tool)

    return anthropic_tools


def _convert_tool_choice(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    parallel_tool_calls = kwargs.pop("parallel_tool_calls", True)
    tool_choice = kwargs.pop("tool_choice", "any")
    if tool_choice == "required":
        tool_choice = "any"
    return {"type": tool_choice, "disable_parallel_tool_use": not parallel_tool_calls}


def _convert_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert kwargs to Anthropic format."""
    kwargs = kwargs.copy()

    if "max_tokens" not in kwargs:
        logger.warning(f"max_tokens is required for Anthropic, setting to {DEFAULT_MAX_TOKENS}")
        kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

    if "tools" in kwargs:
        kwargs["tools"] = _convert_tool_spec(kwargs["tools"])

    if "tool_choice" in kwargs or "parallel_tool_calls" in kwargs:
        kwargs["tool_choice"] = _convert_tool_choice(kwargs)

    return kwargs
