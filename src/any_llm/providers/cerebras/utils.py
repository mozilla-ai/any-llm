from typing import Any, Dict

try:
    from cerebras.cloud.sdk.types.chat.chat_completion import ChatChunkResponse
except ImportError:
    msg = "cerebras is not installed. Please install it with `pip install any-llm-sdk[cerebras]`"
    raise ImportError(msg)

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
)
from any_llm.providers.helpers import create_completion_from_response


def _create_openai_chunk_from_cerebras_chunk(chunk: ChatChunkResponse) -> ChatCompletionChunk:
    """Convert Cerebras streaming chunk to OpenAI ChatCompletionChunk format."""
    if not hasattr(chunk, "choices") or not hasattr(chunk, "model"):
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
            content = getattr(choice_delta, "content", None)
            if content:
                delta["content"] = content

            role = getattr(choice_delta, "role", None)
            if role:
                delta["role"] = role

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
    """Convert Cerebras response using generic helper (already OpenAI-like)."""
    # Straight pass-through with minimal normalization; already OpenAI compliant
    return create_completion_from_response(
        response_data=response_data,
        model=response_data.get("model", ""),
        provider_name="cerebras",
    )
