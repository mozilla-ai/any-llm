from collections.abc import Sequence
from typing import Any

from cohere.types import ListModelsResponse as CohereListModelsResponse

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    Function,
)
from any_llm.types.model import Model


def _patch_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Patches messages for Cohere API compatibility.

    - Removes the 'name' field from tool messages.
    - Converts 'content' to 'tool_plan' in assistant messages with tool_calls.
    - Validates the message sequence.
    """
    patched_messages = []
    for i, message in enumerate(messages):
        patched_message = message.copy()
        if patched_message.get("role") == "tool":
            if i > 0 and messages[i - 1].get("role") != "assistant":
                msg = "A tool message must be preceded by an assistant message with tool_calls."
                raise ValueError(msg)
            patched_message.pop("name", None)
        if patched_message.get("role") == "assistant" and patched_message.get("tool_calls"):
            patched_message["tool_plan"] = patched_message.pop("content")
        patched_messages.append(patched_message)
    return patched_messages


def _create_openai_chunk_from_cohere_chunk(chunk: Any) -> ChatCompletionChunk:
    """Convert Cohere streaming chunk to OpenAI ChatCompletionChunk format."""
    chunk_dict: dict[str, Any] = {
        "id": f"chatcmpl-{hash(str(chunk))}",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "cohere-model",
        "choices": [],
        "usage": None,
    }

    delta: dict[str, Any] = {}
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
    """Convert Cohere response to OpenAI ChatCompletion format directly."""
    prompt_tokens = 0
    completion_tokens = 0

    if response.usage and response.usage.tokens:
        prompt_tokens = int(response.usage.tokens.input_tokens or 0)
        completion_tokens = int(response.usage.tokens.output_tokens or 0)

    usage = CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    if response.finish_reason == "TOOL_CALL" and response.message.tool_calls:
        tool_call = response.message.tool_calls[0]
        message = ChatCompletionMessage(
            role="assistant",
            content=response.message.tool_plan,
            tool_calls=[
                ChatCompletionMessageFunctionToolCall(
                    id=tool_call.id,
                    type="function",
                    function=Function(
                        name=tool_call.function.name if tool_call.function else "",
                        arguments=tool_call.function.arguments if tool_call.function else "",
                    ),
                )
            ],
        )
        choice = Choice(index=0, finish_reason="tool_calls", message=message)
        return ChatCompletion(
            id=getattr(response, "id", ""),
            model=model,
            created=getattr(response, "created", 0),
            object="chat.completion",
            choices=[choice],
            usage=usage,
        )
    content = ""
    if response.message.content and len(response.message.content) > 0:
        content = response.message.content[0].text

    from typing import Literal, cast

    message = ChatCompletionMessage(role=cast("Literal['assistant']", "assistant"), content=content, tool_calls=None)
    choice = Choice(
        index=0,
        finish_reason=cast("Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']", "stop"),
        message=message,
    )
    return ChatCompletion(
        id=getattr(response, "id", ""),
        model=model,
        created=getattr(response, "created", 0),
        object="chat.completion",
        choices=[choice],
        usage=usage,
    )


def _convert_models_list(response: CohereListModelsResponse) -> Sequence[Model]:
    """Converts a Cohere ListModelsResponse to a list of Model objects."""
    models = []
    if response.models:
        for model_data in response.models:
            models.append(
                Model(
                    id=model_data.name or "unknown",
                    created=0,  # Cohere doesn't provide this, so we use a default value
                    object="model",
                    owned_by="cohere",
                )
            )
    return models
