from typing import Any

from cerebras.cloud.sdk.types import ModelListResponse as CerebrasModelListResponse
from cerebras.cloud.sdk.types.chat.chat_completion import ChatChunkResponse

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
from any_llm.types.model import Model


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

    chunk_dict: dict[str, Any] = {
        "id": getattr(chunk, "id", None) or f"chatcmpl-{hash(str(chunk))}",
        "object": "chat.completion.chunk",
        "created": getattr(chunk, "created", None) or 0,
        "model": getattr(chunk, "model", None) or "cerebras-model",
        "choices": [],
        "usage": None,
    }

    delta: dict[str, Any] = {}
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


def _convert_response(response_data: dict[str, Any]) -> ChatCompletion:
    """Convert Cerebras response to OpenAI ChatCompletion directly."""
    choices_out: list[Choice] = []
    for i, choice_data in enumerate(response_data.get("choices", [])):
        message_data = choice_data.get("message", {})
        tool_calls_data = message_data.get("tool_calls") or []
        tool_calls: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] | None = None
        if tool_calls_data:
            tool_calls_list: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] = []
            for tc in tool_calls_data:
                func = tc.get("function", {})
                tool_calls_list.append(
                    ChatCompletionMessageFunctionToolCall(
                        id=tc.get("id", f"call_{i}"),
                        type="function",
                        function=Function(
                            name=func.get("name"),
                            arguments=func.get("arguments"),
                        ),
                    )
                )
            tool_calls = tool_calls_list
        message = ChatCompletionMessage(
            role=message_data.get("role", "assistant"),
            content=message_data.get("content"),
            tool_calls=tool_calls,
        )
        from typing import Literal, cast

        choices_out.append(
            Choice(
                index=choice_data.get("index", i),
                finish_reason=cast(
                    "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']",
                    choice_data.get("finish_reason", "stop"),
                ),
                message=message,
            )
        )

    usage = None
    if response_data.get("usage"):
        usage_raw = response_data["usage"]
        usage = CompletionUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )

    return ChatCompletion(
        id=response_data.get("id", ""),
        model=response_data.get("model", ""),
        created=response_data.get("created", 0),
        object="chat.completion",
        choices=choices_out,
        usage=usage,
    )


def _convert_models_list(models_list: CerebrasModelListResponse) -> list[Model]:
    return [
        Model(id=model.id, object="model", created=model.created or 0, owned_by="cerebras")
        for model in models_list.data
    ]
