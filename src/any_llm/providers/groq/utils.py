from typing import cast, Literal

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from groq.types.chat import ChatCompletionChunk as GroqChatCompletionChunk


def _create_openai_chunk_from_groq_chunk(groq_chunk: GroqChatCompletionChunk) -> ChatCompletionChunk:
    """Convert a Groq streaming chunk to OpenAI ChatCompletionChunk format."""

    choice_data = groq_chunk.choices[0]
    delta_data = choice_data.delta

    delta = ChoiceDelta(
        content=delta_data.content,
        role=cast(Literal["developer", "system", "user", "assistant", "tool"] | None, delta_data.role),
    )

    if delta_data.tool_calls:
        openai_tool_calls = []
        for tool_call in delta_data.tool_calls:
            openai_tool_call = ChoiceDeltaToolCall(
                index=tool_call.index if tool_call.index is not None else 0,
                id=tool_call.id,
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=tool_call.function.name if tool_call.function else None,
                    arguments=tool_call.function.arguments if tool_call.function else None,
                )
                if tool_call.function
                else None,
            )
            openai_tool_calls.append(openai_tool_call)
        delta.tool_calls = openai_tool_calls
    else:
        delta.tool_calls = None

    choice = Choice(
        index=choice_data.index,
        delta=delta,
        finish_reason=choice_data.finish_reason,
    )

    usage = None
    usage_data = groq_chunk.usage
    if usage_data:
        usage = CompletionUsage(
            prompt_tokens=usage_data.prompt_tokens,
            completion_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens,
        )

    return ChatCompletionChunk(
        id=groq_chunk.id,
        choices=[choice],
        created=groq_chunk.created,
        model=groq_chunk.model,
        object="chat.completion.chunk",
        usage=usage,
    )
