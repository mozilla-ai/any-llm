import uuid
from typing import Literal, cast

from huggingface_hub.inference._generated.types import (  # type: ignore[attr-defined]
    ChatCompletionStreamOutput as HuggingFaceChatCompletionStreamOutput,
)

from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
)


def _create_openai_chunk_from_huggingface_chunk(chunk: HuggingFaceChatCompletionStreamOutput) -> ChatCompletionChunk:
    """Convert a HuggingFace streaming chunk to OpenAI ChatCompletionChunk format."""

    chunk_id = f"chatcmpl-{uuid.uuid4()}"
    created = chunk.created
    model = chunk.model

    choices = []
    hf_choices = chunk.choices

    for i, hf_choice in enumerate(hf_choices):
        hf_delta = hf_choice.delta
        content = hf_delta.content
        role = hf_delta.role

        openai_role = None
        if role:
            openai_role = cast("Literal['developer', 'system', 'user', 'assistant', 'tool']", role)

        delta = ChoiceDelta(content=content, role=openai_role)

        choice = ChunkChoice(
            index=i,
            delta=delta,
            finish_reason=cast(
                "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'] | None",
                hf_choice.finish_reason,
            ),
        )
        choices.append(choice)

    usage = None
    hf_usage = chunk.usage
    if hf_usage:
        prompt_tokens = hf_usage.prompt_tokens
        completion_tokens = hf_usage.completion_tokens
        total_tokens = hf_usage.total_tokens

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
