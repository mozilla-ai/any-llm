import uuid
from typing import Literal, cast
from datetime import datetime

from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    CompletionUsage,
    ChunkChoice,
)
from together.types.chat_completions import ChatCompletionChunk as TogetherChatCompletionChunk


def _create_openai_chunk_from_together_chunk(together_chunk: TogetherChatCompletionChunk) -> ChatCompletionChunk:
    """Convert a Together streaming chunk to OpenAI ChatCompletionChunk format."""

    if not together_chunk.choices:
        raise ValueError("Together chunk has no choices")

    together_choice = together_chunk.choices[0]
    delta_content = together_choice.delta
    if not delta_content:
        raise ValueError("Together chunk has no delta")

    content = delta_content.content
    role = None
    if delta_content.role:  # type: ignore[attr-defined]
        role = cast(Literal["assistant", "user", "system"], delta_content.role)  # type: ignore[attr-defined]

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
            Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None,
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
