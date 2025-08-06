import uuid
from typing import Any
from datetime import datetime

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)
from openai.types.completion_usage import CompletionUsage


def _create_openai_chunk_from_fireworks_chunk(fireworks_chunk: Any) -> ChatCompletionChunk:
    """Convert a Fireworks streaming chunk to OpenAI ChatCompletionChunk format."""

    content = None
    if hasattr(fireworks_chunk, "choices") and fireworks_chunk.choices:
        choice = fireworks_chunk.choices[0]
        if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
            content = choice.delta.content
        elif hasattr(choice, "text"):
            content = choice.text

    delta = ChoiceDelta(content=content)

    finish_reason = None
    if hasattr(fireworks_chunk, "choices") and fireworks_chunk.choices:
        choice = fireworks_chunk.choices[0]
        if hasattr(choice, "finish_reason"):
            finish_reason = choice.finish_reason

    choice_obj = Choice(
        index=0,
        delta=delta,
        finish_reason=finish_reason,
    )

    usage = None
    if hasattr(fireworks_chunk, "usage") and fireworks_chunk.usage:
        usage_data = fireworks_chunk.usage
        usage = CompletionUsage(
            prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
            completion_tokens=getattr(usage_data, "completion_tokens", 0),
            total_tokens=getattr(usage_data, "total_tokens", 0),
        )

    created = int(datetime.now().timestamp())

    return ChatCompletionChunk(
        id=f"chatcmpl-{uuid.uuid4()}",
        choices=[choice_obj],
        created=created,
        model=getattr(fireworks_chunk, "model", "unknown"),
        object="chat.completion.chunk",
        usage=usage,
    )
