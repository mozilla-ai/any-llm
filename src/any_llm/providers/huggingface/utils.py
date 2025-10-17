import re
import uuid
from collections.abc import Iterable
from typing import Any, Literal, cast

from huggingface_hub.hf_api import ModelInfo as HfModelInfo
from huggingface_hub.inference._generated.types import (  # type: ignore[attr-defined]
    ChatCompletionStreamOutput as HuggingFaceChatCompletionStreamOutput,
)
from openai.lib._parsing import type_to_response_format_param

from any_llm.constants import REASONING_FIELD_NAMES
from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChunkChoice,
    CompletionParams,
    CompletionUsage,
    Reasoning,
)
from any_llm.types.model import Model


def _normalize_reasoning_on_message(message_dict: dict[str, Any]) -> None:
    """Mutate a message dict to extract reasoning from content tags and provider-specific fields."""
    if isinstance(message_dict.get("reasoning"), dict) and "content" in message_dict["reasoning"]:
        return

    reasoning_content = None

    for field_name in REASONING_FIELD_NAMES:
        if field_name in message_dict and message_dict[field_name] is not None:
            reasoning_content = message_dict[field_name]
            break

    if reasoning_content is None and isinstance(message_dict.get("reasoning"), str):
        reasoning_content = message_dict["reasoning"]

    content = message_dict.get("content")
    if isinstance(content, str):
        for tag_name in REASONING_FIELD_NAMES:
            tag_open = f"<{tag_name}>"
            tag_close = f"</{tag_name}>"
            think_pattern = re.escape(tag_open) + r"(.*?)" + re.escape(tag_close)
            matches = re.findall(think_pattern, content, re.DOTALL)
            if matches:
                extracted_reasoning = "\n".join(matches)
                if reasoning_content:
                    reasoning_content = f"{reasoning_content}\n{extracted_reasoning}"
                else:
                    reasoning_content = extracted_reasoning
                content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()

        message_dict["content"] = content

    if reasoning_content is not None:
        message_dict["reasoning"] = {"content": str(reasoning_content)}


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

        _normalize_reasoning_on_message(delta_dict)

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
