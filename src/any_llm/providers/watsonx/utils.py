from typing import Any, Literal, cast
import uuid

from any_llm.types.completion import (
    ChatCompletion,
    Choice,
    ChatCompletionChunk,
    ChoiceDelta,
    CompletionUsage,
    ChatCompletionMessage,
    ChunkChoice,
)


def _convert_response(response: dict[str, Any]) -> ChatCompletion:
    """Convert Watsonx response to OpenAI ChatCompletion format."""
    choice_data = response["choices"][0]
    message_data = choice_data["message"]

    message = ChatCompletionMessage(
        content=message_data.get("content"),
        role=message_data.get("role", "assistant"),
        tool_calls=None,  # Watsonx doesn't seem to support tool calls in the aisuite implementation
    )

    choice = Choice(
        finish_reason=choice_data.get("finish_reason", "stop"),
        index=choice_data.get("index", 0),
        message=message,
    )

    usage = None
    if "usage" in response:
        usage_data = response["usage"]
        usage = CompletionUsage(
            completion_tokens=usage_data.get("completion_tokens", 0),
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

    # Build the final ChatCompletion object
    return ChatCompletion(
        id=response.get("id", ""),
        model=response.get("model", ""),
        object="chat.completion",
        created=response.get("created", 0),
        choices=[choice],
        usage=usage,
    )


def _convert_streaming_chunk(chunk: dict[str, Any]) -> ChatCompletionChunk:
    """Convert Watsonx streaming chunk to OpenAI ChatCompletionChunk format."""
    chunk_id = f"chatcmpl-{uuid.uuid4()}"
    created = chunk.get("created", 0)
    model = chunk.get("model", "")

    choices = []
    chunk_choices = chunk.get("choices", [])

    for i, chunk_choice in enumerate(chunk_choices):
        delta_data = chunk_choice.get("delta", {})
        content = delta_data.get("content")
        role = delta_data.get("role")

        openai_role = None
        if role:
            openai_role = cast(Literal["developer", "system", "user", "assistant", "tool"], role)

        delta = ChoiceDelta(content=content, role=openai_role)

        choice = ChunkChoice(
            index=i,
            delta=delta,
            finish_reason=cast(
                Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None,
                chunk_choice.get("finish_reason"),
            ),
        )
        choices.append(choice)

    usage = None
    chunk_usage = chunk.get("usage")
    if chunk_usage:
        prompt_tokens = chunk_usage.get("prompt_tokens", 0)
        completion_tokens = chunk_usage.get("completion_tokens", 0)
        total_tokens = chunk_usage.get("total_tokens", 0)

        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    return ChatCompletionChunk(
        id=chunk_id,
        choices=choices,
        created=created,
        model=model,
        object="chat.completion.chunk",
        usage=usage,
    )
