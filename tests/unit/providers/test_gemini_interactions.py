import asyncio
import json
import logging
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from google.genai import types
from google.genai._gaos.types.interactions import Interaction
from google.genai.interactions import (
    ArgumentsDelta,
    Error,
    ErrorEvent,
    InteractionCompletedEvent,
    InteractionCreatedEvent,
    InteractionSSEEvent,
    InteractionSseEventInteraction,
    InteractionSseEventInteractionStatus,
    InteractionStatusUpdate,
    ModelOutputStep,
    Step,
    StepDelta,
    StepStart,
    StepStop,
    TextContent,
    TextDelta,
    UnknownInteractionSSEEvent,
    UnknownStepDeltaData,
    Usage,
    UserInputStep,
)
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from any_llm.exceptions import InvalidRequestError, ProviderError, UnsupportedParameterError
from any_llm.providers.gemini import GeminiProvider
from any_llm.providers.gemini.base import GoogleProvider
from any_llm.providers.gemini.interactions import (
    convert_interaction_to_response,
    convert_responses_params,
)
from any_llm.providers.gemini.interactions_stream import convert_interaction_stream
from any_llm.providers.vertexai import VertexaiProvider
from any_llm.types.responses import Response, ResponsesParams, ResponseStreamEvent


def _interaction(
    *,
    status: str = "completed",
    created: str = "2026-01-02T03:04:05Z",
    steps: list[object] | None = None,
    usage: Usage | None = None,
) -> Interaction:
    if steps is None:
        steps = [ModelOutputStep(content=[TextContent(text="Hello")])]
    if usage is None:
        usage = Usage(
            total_input_tokens=4,
            total_output_tokens=2,
            total_tokens=6,
            total_cached_tokens=1,
            total_thought_tokens=3,
        )
    return Interaction.model_validate(
        {
            "id": "int-123",
            "status": status,
            "model": "gemini-3.8-flash",
            "created": created,
            "previous_interaction_id": "int-previous",
            "system_instruction": "Be concise",
            "labels": {"team": "sdk"},
            "steps": steps,
            "usage": usage,
        }
    )


async def _events(*events: InteractionSSEEvent) -> AsyncIterator[InteractionSSEEvent]:
    for event in events:
        yield event


def _created(*, model: str | None = None) -> InteractionCreatedEvent:
    return InteractionCreatedEvent(
        interaction=InteractionSseEventInteraction(
            id="int-123",
            status="in_progress",
            model=model,
        )
    )


def _completed(
    status: InteractionSseEventInteractionStatus = "completed",
    *,
    model: str | None = None,
    steps: list[Step] | None = None,
) -> InteractionCompletedEvent:
    return InteractionCompletedEvent(
        interaction=InteractionSseEventInteraction(
            id="int-123",
            status=status,
            model=model,
            steps=steps,
        )
    )


async def _converted_events(*events: InteractionSSEEvent, model: str = "requested") -> list[ResponseStreamEvent]:
    return [event async for event in convert_interaction_stream(_events(*events), model=model)]


def test_gemini_enables_responses_without_changing_shared_google_provider() -> None:
    assert GeminiProvider.SUPPORTS_RESPONSES is True
    assert GoogleProvider.SUPPORTS_RESPONSES is False
    assert VertexaiProvider.SUPPORTS_RESPONSES is False


def test_convert_interaction_maps_text_status_metadata_and_usage() -> None:
    response = convert_interaction_to_response(_interaction())

    assert isinstance(response, Response)
    assert response.id == "int-123"
    assert response.status == "completed"
    assert response.model == "gemini-3.8-flash"
    assert response.created_at == 1767323045.0
    assert response.previous_response_id == "int-previous"
    assert response.instructions == "Be concise"
    assert response.metadata == {"team": "sdk"}
    assert response.output_text == "Hello"
    message = response.output[0]
    assert isinstance(message, ResponseOutputMessage)
    assert message.id == "msg-0"
    assert response.usage is not None
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 2
    assert response.usage.total_tokens == 6
    assert response.usage.input_tokens_details.cached_tokens == 1
    assert response.usage.output_tokens_details.reasoning_tokens == 3


def test_convert_interaction_preserves_explicit_zero_total_usage() -> None:
    usage = Usage(total_input_tokens=2, total_output_tokens=3, total_tokens=0)
    response = convert_interaction_to_response(_interaction(usage=usage))

    assert response.usage is not None
    assert response.usage.total_tokens == 0


def test_convert_interaction_preserves_empty_text_output() -> None:
    response = convert_interaction_to_response(_interaction(steps=[ModelOutputStep(content=[TextContent(text="")])]))

    assert len(response.output) == 1
    assert response.output_text == ""


def test_convert_interaction_skips_unsupported_steps_and_content() -> None:
    interaction = _interaction(
        steps=[
            {"type": "future_step", "future": True},
            ModelOutputStep.model_validate({"content": [{"type": "future_content", "future": True}]}),
            ModelOutputStep.model_validate(
                {
                    "content": [
                        {"type": "future_content", "future": True},
                        {"type": "text", "text": "kept"},
                    ]
                }
            ),
        ]
    )

    response = convert_interaction_to_response(interaction)

    assert response.output_text == "kept"
    assert len(response.output) == 1
    assert response.output[0].id == "msg-0"


def test_convert_interaction_maps_provider_error_without_raw_side_channel() -> None:
    interaction = _interaction(status="failed", steps=[])
    interaction.errors = [Error(code="gateway_timeout", message="deadline expired")]

    response = convert_interaction_to_response(interaction)

    assert response.error is not None
    assert response.error.code == "server_error"
    assert response.error.message == "deadline expired"
    assert not any(name.startswith("gemini_") for name in response.model_dump())


def test_convert_interaction_handles_unknown_status_and_invalid_timestamp() -> None:
    response = convert_interaction_to_response(_interaction(status="future_status", created="invalid"))

    assert response.status == "in_progress"
    assert response.created_at == 0.0


def test_convert_responses_params_maps_only_reviewed_text_subset() -> None:
    params = ResponsesParams(
        model="gemini-3.8-flash",
        input="Hello",
        instructions="",
        max_output_tokens=0,
        stream=True,
    )

    assert convert_responses_params(params, "gemini", api_version="v1") == {
        "api_version": "v1",
        "model": "gemini-3.8-flash",
        "input": "Hello",
        "system_instruction": "",
        "generation_config": {"max_output_tokens": 0},
        "stream": True,
    }


def test_convert_responses_params_rejects_non_string_input() -> None:
    params = ResponsesParams(model="gemini-3.8-flash", input=[{"type": "input_text", "text": "Hello"}])

    with pytest.raises(UnsupportedParameterError, match="input"):
        convert_responses_params(params, "gemini", api_version="v1")


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("tools", [{"type": "function", "name": "lookup"}]),
        ("reasoning", {"effort": "low"}),
        ("response_format", {"type": "json_object"}),
        ("background", True),
        ("temperature", 0.2),
        ("store", False),
        ("metadata", {}),
        ("previous_response_id", "int-previous"),
    ],
)
def test_convert_responses_params_rejects_unimplemented_surface(parameter: str, value: object) -> None:
    params = ResponsesParams.model_validate({"model": "gemini-3.8-flash", "input": "Hello", parameter: value})

    with pytest.raises(UnsupportedParameterError, match=parameter):
        convert_responses_params(params, "gemini", api_version="v1")
