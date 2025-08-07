from typing import Any, Dict

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.providers.helpers import create_completion_from_response


def _create_openai_chunk_from_cohere_chunk(chunk: Any) -> ChatCompletionChunk:
    """Convert Cohere streaming chunk to OpenAI ChatCompletionChunk format."""
    # Default chunk structure
    chunk_dict: Dict[str, Any] = {
        "id": f"chatcmpl-{hash(str(chunk))}",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "cohere-model",
        "choices": [],
        "usage": None,
    }

    delta: Dict[str, Any] = {}
    finish_reason = None

    chunk_type = getattr(chunk, "type", None)

    if chunk_type == "content-delta":
        if (
            hasattr(chunk, "delta")
            and chunk.delta
            and hasattr(chunk.delta, "message")
            and chunk.delta.message
            and hasattr(chunk.delta.message, "content")
            and chunk.delta.message.content
            and hasattr(chunk.delta.message.content, "text")
        ):
            delta["content"] = chunk.delta.message.content.text

    elif chunk_type == "tool-call-start":
        if (
            hasattr(chunk, "delta")
            and chunk.delta
            and hasattr(chunk.delta, "message")
            and chunk.delta.message
            and hasattr(chunk.delta.message, "tool_calls")
            and chunk.delta.message.tool_calls
        ):
            tool_call = chunk.delta.message.tool_calls
            delta["tool_calls"] = [
                {
                    "index": 0,
                    "id": getattr(tool_call, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(tool_call.function, "name", "")
                        if hasattr(tool_call, "function") and tool_call.function
                        else "",
                        "arguments": "",
                    },
                }
            ]

    elif chunk_type == "tool-call-delta":
        if (
            hasattr(chunk, "delta")
            and chunk.delta
            and hasattr(chunk.delta, "message")
            and chunk.delta.message
            and hasattr(chunk.delta.message, "tool_calls")
            and chunk.delta.message.tool_calls
            and hasattr(chunk.delta.message.tool_calls, "function")
            and chunk.delta.message.tool_calls.function
        ):
            delta["tool_calls"] = [
                {
                    "index": 0,
                    "function": {
                        "arguments": getattr(chunk.delta.message.tool_calls.function, "arguments", ""),
                    },
                }
            ]

    elif chunk_type == "tool-call-end":
        finish_reason = "tool_calls"

    elif chunk_type == "message-end":
        finish_reason = "stop"

        if (
            hasattr(chunk, "delta")
            and chunk.delta
            and hasattr(chunk.delta, "usage")
            and chunk.delta.usage
            and hasattr(chunk.delta.usage, "tokens")
            and chunk.delta.usage.tokens
        ):
            chunk_dict["usage"] = {
                "prompt_tokens": int(getattr(chunk.delta.usage.tokens, "input_tokens", 0) or 0),
                "completion_tokens": int(getattr(chunk.delta.usage.tokens, "output_tokens", 0) or 0),
                "total_tokens": int(
                    (getattr(chunk.delta.usage.tokens, "input_tokens", 0) or 0)
                    + (getattr(chunk.delta.usage.tokens, "output_tokens", 0) or 0)
                ),
            }

    choice_dict = {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
        "logprobs": None,
    }

    chunk_dict["choices"] = [choice_dict]

    return ChatCompletionChunk.model_validate(chunk_dict)


def _convert_response(response: Any, model: str) -> ChatCompletion:
    """Convert Cohere response to OpenAI ChatCompletion format."""
    prompt_tokens = 0
    completion_tokens = 0

    if response.usage and response.usage.tokens:
        prompt_tokens = int(response.usage.tokens.input_tokens or 0)
        completion_tokens = int(response.usage.tokens.output_tokens or 0)

    response_dict = {
        "id": getattr(response, "id", ""),
        "model": getattr(response, "model", ""),
        "created": getattr(response, "created", 0),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    if response.finish_reason == "TOOL_CALL" and response.message.tool_calls:
        tool_call = response.message.tool_calls[0]
        response_dict["choices"] = [
            {
                "message": {
                    "role": "assistant",
                    "content": response.message.tool_plan,  # Use tool_plan as content
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "function": {
                                "name": tool_call.function.name if tool_call.function else "",
                                "arguments": tool_call.function.arguments if tool_call.function else "",
                            },
                            "type": "function",
                        }
                    ],
                },
                "finish_reason": "tool_calls",
                "index": 0,
            }
        ]
    else:
        content = ""
        if response.message.content and len(response.message.content) > 0:
            content = response.message.content[0].text

        response_dict["choices"] = [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ]

    return create_completion_from_response(
        response_data=response_dict,
        model=model,
        provider_name="cohere",
    )
