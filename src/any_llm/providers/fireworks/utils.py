import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from openai.pagination import SyncPage
from openai.types.model import Model as OpenAIModel

from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
)
from any_llm.types.model import Model


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

    choice_obj = ChunkChoice(
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


def _convert_models_list(
    fireworksai_models: SyncPage[OpenAIModel],
    provider_name: str,
) -> Sequence[Model]:
    """
    Convert OpenAI models list to OpenAI Model format.
    Each openai_model can be a dict or a pydantic BaseModel.
    """
    results = []
    for fireworksai_model in fireworksai_models:
        model = Model(
            id=fireworksai_model.id,
            label=fireworksai_model.id,
            created=fireworksai_model.created,
            object=fireworksai_model.object or "model",
            provider=provider_name,
            owned_by=fireworksai_model.owned_by or provider_name,
            attributes=fireworksai_model.model_dump(),
        )
        results.append(model)
    return results
