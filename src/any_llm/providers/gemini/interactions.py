from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from google.genai.interactions import (
    InteractionSseEventInteraction,
    ModelOutputStep,
    TextContent,
    Usage,
)
from openai.types.responses import (
    Response as OpenAIResponse,
)
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseStatus,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from any_llm.exceptions import UnsupportedParameterError

if TYPE_CHECKING:
    from collections.abc import Sequence

    # google-genai's public re-export is correct at runtime, but mypy resolves
    # Interaction to its generated request union. Keep the private import type-only.
    # https://github.com/googleapis/python-genai/blob/v2.22.0/google/genai/interactions.py
    from google.genai._gaos.types.interactions import Interaction

    from any_llm.types.responses import ResponsesParams


# The stable v1 schema has six statuses. google-genai 2.17.0 also accepts
# queued and budget_exceeded, so normalize those SDK values without exposing
# them through the narrower OpenAI Response status type.
# https://ai.google.dev/api/interactions-api-v1#Interaction
_STATUS_MAP: dict[str, ResponseStatus] = {
    "completed": "completed",
    "failed": "failed",
    "in_progress": "in_progress",
    "cancelled": "cancelled",
    "queued": "queued",
    "incomplete": "incomplete",
    "requires_action": "incomplete",
    "budget_exceeded": "incomplete",
}


def _iso_to_epoch(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _map_status(status: object) -> ResponseStatus:
    return _STATUS_MAP.get(str(status), "in_progress")


def _convert_usage(usage: Usage | None) -> ResponseUsage | None:
    if usage is None:
        return None
    input_tokens = usage.total_input_tokens or 0
    output_tokens = usage.total_output_tokens or 0
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=InputTokensDetails(
            cached_tokens=usage.total_cached_tokens or 0,
            cache_write_tokens=0,
        ),
        output_tokens=output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=usage.total_thought_tokens or 0),
        total_tokens=usage.total_tokens if usage.total_tokens is not None else input_tokens + output_tokens,
    )


def _message_status(status: ResponseStatus) -> Literal["completed", "in_progress", "incomplete"]:
    if status == "completed":
        return "completed"
    if status in {"in_progress", "queued"}:
        return "in_progress"
    return "incomplete"


def _messages_from_steps(
    steps: Sequence[object] | None,
    status: ResponseStatus,
) -> list[ResponseOutputMessage]:
    messages: list[ResponseOutputMessage] = []
    for step in steps or []:
        if not isinstance(step, ModelOutputStep):
            continue
        text_parts = [part.text for part in step.content or [] if isinstance(part, TextContent)]
        if not text_parts:
            continue
        text = "".join(text_parts)
        output_index = len(messages)
        messages.append(
            ResponseOutputMessage(
                id=f"msg-{output_index}",
                type="message",
                role="assistant",
                status=_message_status(status),
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        )
    return messages


def _response_from_interaction(
    interaction: Interaction | InteractionSseEventInteraction,
    *,
    fallback_model: str = "",
) -> OpenAIResponse:
    status = _map_status(interaction.status)
    previous_response_id = None
    instructions = None
    metadata = None
    response_error = None
    if not isinstance(interaction, InteractionSseEventInteraction):
        previous_response_id = interaction.previous_interaction_id
        instructions = interaction.system_instruction
        metadata = interaction.labels
        if interaction.errors:
            first_error = interaction.errors[0]
            response_error = {
                "code": "server_error",
                "message": first_error.message or first_error.code or "Gemini interaction failed",
            }

    return OpenAIResponse.model_validate(
        {
            "id": interaction.id or "",
            "created_at": _iso_to_epoch(interaction.created),
            "error": response_error,
            "instructions": instructions,
            "metadata": metadata,
            "model": str(interaction.model or fallback_model),
            "object": "response",
            "output": _messages_from_steps(interaction.steps, status),
            "parallel_tool_calls": False,
            "status": status,
            "tool_choice": "auto",
            "tools": [],
            "previous_response_id": previous_response_id,
            "usage": _convert_usage(interaction.usage),
        }
    )


def convert_interaction_to_response(interaction: Interaction) -> OpenAIResponse:
    """Normalize the text subset of a Gemini Interaction resource."""
    return _response_from_interaction(interaction)


def convert_responses_params(
    params: ResponsesParams,
    provider_name: str,
    *,
    api_version: str,
) -> dict[str, Any]:
    """Translate the supported Responses subset into Interactions arguments."""
    if not isinstance(params.input, str):
        parameter_name = "input"
        raise UnsupportedParameterError(parameter_name, provider_name)

    supported = {
        "model",
        "input",
        "instructions",
        "max_output_tokens",
        "stream",
    }
    unsupported = params.model_dump(exclude_none=True).keys() - supported
    if unsupported:
        raise UnsupportedParameterError(min(unsupported), provider_name)

    # The SDK has separate streaming and non-streaming overloads, while optional
    # fields must be absent rather than None. Any stays at this generated SDK
    # boundary because object values cannot satisfy either unpacked overload.
    create_kwargs: dict[str, Any] = {
        "api_version": api_version,
        "model": params.model,
        "input": params.input,
    }
    if params.instructions is not None:
        create_kwargs["system_instruction"] = params.instructions
    if params.max_output_tokens is not None:
        create_kwargs["generation_config"] = {"max_output_tokens": params.max_output_tokens}
    if params.stream:
        create_kwargs["stream"] = True
    return create_kwargs
