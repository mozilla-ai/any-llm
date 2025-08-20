import uuid
from collections.abc import Iterable
from typing import Any, Literal, cast

from huggingface_hub.hf_api import ModelInfo as HfModelInfo
from huggingface_hub.inference._generated.types import (  # type: ignore[attr-defined]
    ChatCompletionStreamOutput as HuggingFaceChatCompletionStreamOutput,
)
from openai.lib._parsing import type_to_response_format_param

from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChunkChoice,
    CompletionParams,
    CompletionUsage,
)
from any_llm.types.model import Model


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


def _convert_params(params: CompletionParams, **kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert CompletionParams to a dictionary of parameters for HuggingFace API."""

    result_kwargs: dict[str, Any] = kwargs.copy()

    # timeout is passed to the client instantiation, should not reach the `client.chat_completion` call.
    result_kwargs.pop("timeout", None)

    if params.max_tokens is not None:
        result_kwargs["max_new_tokens"] = params.max_tokens

    if params.reasoning_effort == "auto":
        params.reasoning_effort = None

    if params.response_format is not None:
        result_kwargs["response_format"] = type_to_response_format_param(response_format=params.response_format)  # type: ignore[arg-type]

    result_kwargs.update(
        params.model_dump(
            exclude_none=True,
            exclude={"max_tokens", "model_id", "messages", "response_format", "parallel_tool_calls"},
        )
    )

    result_kwargs["model"] = params.model_id
    result_kwargs["messages"] = params.messages

    return result_kwargs


def _convert_models_list(models_list: Iterable[HfModelInfo]) -> list[Model]:
    return [
        Model(
            id=model.id,
            object="model",
            created=int(model.created_at.timestamp()) if model.created_at else 0,
            owned_by="huggingface",
        )
        for model in models_list
    ]
