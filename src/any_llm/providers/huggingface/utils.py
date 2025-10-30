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
    Reasoning,
)
from any_llm.types.model import Model
from any_llm.utils.reasoning import normalize_reasoning_from_provider_fields_and_xml_tags


def _create_openai_chunk_from_huggingface_chunk(chunk: HuggingFaceChatCompletionStreamOutput) -> ChatCompletionChunk:
    """Convert a HuggingFace streaming chunk to OpenAI ChatCompletionChunk format."""

    chunk_id = f"chatcmpl-{uuid.uuid4()}"
    created = chunk.created
    model = chunk.model

    choices = []
    hf_choices = chunk.choices

    for i, hf_choice in enumerate(hf_choices):
        hf_delta = hf_choice.delta

        delta_dict: dict[str, Any] = {}
        if hf_delta.content is not None:
            delta_dict["content"] = hf_delta.content
        if hf_delta.role is not None:
            delta_dict["role"] = hf_delta.role
        if hasattr(hf_delta, "reasoning"):
            delta_dict["reasoning"] = hf_delta.reasoning

        normalize_reasoning_from_provider_fields_and_xml_tags(delta_dict)

        openai_role = None
        if delta_dict.get("role"):
            openai_role = cast("Literal['developer', 'system', 'user', 'assistant', 'tool']", delta_dict["role"])

        reasoning_obj = None
        if delta_dict.get("reasoning") and isinstance(delta_dict["reasoning"], dict):
            if "content" in delta_dict["reasoning"]:
                reasoning_obj = Reasoning(content=delta_dict["reasoning"]["content"])

        delta = ChoiceDelta(
            content=delta_dict.get("content"),
            role=openai_role,
            reasoning=reasoning_obj,
        )

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

    if params.reasoning_effort == "auto":
        params.reasoning_effort = None

    if params.response_format is not None:
        result_kwargs["response_format"] = type_to_response_format_param(response_format=params.response_format)  # type: ignore[arg-type]

    result_kwargs.update(
        params.model_dump(
            exclude_none=True,
            exclude={"model_id", "messages", "response_format", "parallel_tool_calls"},
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
