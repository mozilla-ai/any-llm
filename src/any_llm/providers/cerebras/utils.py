from typing import Any, Dict

try:
    from cerebras.cloud.sdk.types.chat.chat_completion import ChatChunkResponse
except ImportError:
    msg = "cerebras is not installed. Please install it with `pip install any-llm-sdk[cerebras]`"
    raise ImportError(msg)

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionUsage,
    ChatCompletionMessage,
    Function,
)
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall


def _create_openai_chunk_from_cerebras_chunk(chunk: ChatChunkResponse) -> ChatCompletionChunk:
    """Convert Cerebras streaming chunk to OpenAI ChatCompletionChunk format."""
    # Handle different chunk types gracefully
    if not hasattr(chunk, "choices") or not hasattr(chunk, "model"):
        # Return empty chunk for unsupported types
        return ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-empty",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "cerebras-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": None, "logprobs": None}],
                "usage": None,
            }
        )

    # Default chunk structure
    chunk_dict: Dict[str, Any] = {
        "id": getattr(chunk, "id", None) or f"chatcmpl-{hash(str(chunk))}",
        "object": "chat.completion.chunk",
        "created": getattr(chunk, "created", None) or 0,
        "model": getattr(chunk, "model", None) or "cerebras-model",
        "choices": [],
        "usage": None,
    }

    delta: Dict[str, Any] = {}
    finish_reason = None

    choices = getattr(chunk, "choices", None)
    if choices and len(choices) > 0:
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)

        choice_delta = getattr(choice, "delta", None)
        if choice_delta:
            # Handle content delta
            content = getattr(choice_delta, "content", None)
            if content:
                delta["content"] = content

            # Handle role delta
            role = getattr(choice_delta, "role", None)
            if role:
                delta["role"] = role

            # Handle tool calls delta
            tool_calls = getattr(choice_delta, "tool_calls", None)
            if tool_calls:
                tool_calls_list = []
                for tool_call in tool_calls:
                    tool_call_dict = {
                        "index": getattr(tool_call, "index", None) or 0,
                        "id": getattr(tool_call, "id", None),
                        "type": getattr(tool_call, "type", None) or "function",
                    }
                    function = getattr(tool_call, "function", None)
                    if function:
                        tool_call_dict["function"] = {
                            "name": getattr(function, "name", None),
                            "arguments": getattr(function, "arguments", None),
                        }
                    tool_calls_list.append(tool_call_dict)
                delta["tool_calls"] = tool_calls_list

    # Add usage info if available
    usage = getattr(chunk, "usage", None)
    if usage:
        chunk_dict["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", None) or 0,
            "total_tokens": getattr(usage, "total_tokens", None) or 0,
        }

    choice_dict = {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
        "logprobs": None,
    }

    chunk_dict["choices"] = [choice_dict]

    return ChatCompletionChunk.model_validate(chunk_dict)


def _convert_response(response_data: Dict[str, Any]) -> ChatCompletion:
    """Convert Cerebras response to OpenAI ChatCompletion format."""
    # Since Cerebras is OpenAI-compliant, the response should already be in the right format
    # We just need to create proper OpenAI objects

    choice_data = response_data["choices"][0]
    message_data = choice_data["message"]

    # Handle tool calls if present
    tool_calls = None
    if "tool_calls" in message_data and message_data["tool_calls"]:
        tool_calls = []
        for tool_call in message_data["tool_calls"]:
            tool_calls.append(
                ChatCompletionMessageFunctionToolCall(
                    id=tool_call.get("id"),
                    type="function",  # Always set to "function" as it's the only valid value
                    function=Function(
                        name=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"],
                    ),
                )
            )

    # Create the message
    message = ChatCompletionMessage(
        content=message_data.get("content"),
        role=message_data.get("role", "assistant"),
        tool_calls=tool_calls,  # type: ignore[arg-type]
    )

    # Create the choice
    from any_llm.types.completion import Choice

    choice = Choice(
        finish_reason=choice_data.get("finish_reason", "stop"),
        index=choice_data.get("index", 0),
        message=message,
    )

    # Create usage information (if available)
    usage = None
    if "usage" in response_data:
        usage_data = response_data["usage"]
        usage = CompletionUsage(
            completion_tokens=usage_data.get("completion_tokens", 0),
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

    # Build the final ChatCompletion object
    return ChatCompletion(
        id=response_data.get("id", ""),
        model=response_data.get("model", ""),
        object="chat.completion",
        created=response_data.get("created", 0),
        choices=[choice],
        usage=usage,
    )
