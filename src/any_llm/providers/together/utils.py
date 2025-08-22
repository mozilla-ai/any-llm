import uuid
from datetime import datetime
from typing import Any, Literal, cast

from together.types.chat_completions import ChatCompletionChunk as TogetherChatCompletionChunk

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    CompletionUsage,
)


def _create_openai_chunk_from_together_chunk(together_chunk: TogetherChatCompletionChunk) -> ChatCompletionChunk:
    """Convert a Together streaming chunk to OpenAI ChatCompletionChunk format."""

    if not together_chunk.choices:
        msg = "Together chunk has no choices"
        raise ValueError(msg)

    together_choice = together_chunk.choices[0]
    delta_content = together_choice.delta
    if not delta_content:
        msg = "Together chunk has no delta"
        raise ValueError(msg)

    content = delta_content.content
    role = None
    if delta_content.role:  # type: ignore[attr-defined]
        role = cast("Literal['assistant', 'user', 'system']", delta_content.role)  # type: ignore[attr-defined]

    delta = ChoiceDelta(content=content, role=role)

    if delta_content.tool_calls:  # type: ignore[attr-defined]
        openai_tool_calls = []
        for tool_call in delta_content.tool_calls:  # type: ignore[attr-defined]
            openai_tool_call = ChoiceDeltaToolCall(
                index=0,
                id=str(uuid.uuid4()),
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
            openai_tool_calls.append(openai_tool_call)
        delta.tool_calls = openai_tool_calls

    choice = ChunkChoice(
        index=0,
        delta=delta,
        finish_reason=cast(
            "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'] | None",
            together_choice.finish_reason,
        ),
    )

    usage = None
    if together_chunk.usage:
        usage = CompletionUsage(
            prompt_tokens=together_chunk.usage.prompt_tokens or 0,
            completion_tokens=together_chunk.usage.completion_tokens or 0,
            total_tokens=together_chunk.usage.total_tokens or 0,
        )

    return ChatCompletionChunk(
        id=together_chunk.id or f"chatcmpl-{uuid.uuid4()}",
        choices=[choice],
        created=together_chunk.created or int(datetime.now().timestamp()),
        model=together_chunk.model or "unknown",
        object="chat.completion.chunk",
        usage=usage,
    )


def _convert_together_response_to_chat_completion(response_data: dict[str, Any], model_id: str) -> ChatCompletion:
    """Convert Together API response to OpenAI ChatCompletion format."""
    choices_out: list[Choice] = []
    for i, ch in enumerate(response_data.get("choices", [])):
        msg = ch.get("message", {})

        message = ChatCompletionMessage(
            role=cast("Literal['assistant']", msg.get("role")),
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls"),
        )
        choices_out.append(
            Choice(
                index=i,
                finish_reason=cast(
                    "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']",
                    ch.get("finish_reason"),
                ),
                message=message,
            )
        )

    usage = None
    if response_data.get("usage"):
        u = response_data["usage"]
        usage = CompletionUsage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )

    return ChatCompletion(
        id=response_data.get("id", ""),
        model=model_id,
        created=response_data.get("created", 0),
        object="chat.completion",
        choices=choices_out,
        usage=usage,
    )
