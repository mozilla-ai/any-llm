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


@pytest.mark.asyncio
async def test_convert_interaction_stream_maps_text_and_terminal_snapshot() -> None:
    started = StepStart(index=0, step=ModelOutputStep(content=[TextContent(text="Hello")]))
    delta = StepDelta(index=0, delta=TextDelta(text=" world"))
    status = InteractionStatusUpdate(interaction_id="int-123", status="in_progress")
    stopped = StepStop(index=0)

    result = await _converted_events(
        _created(model="gemini-3.8-flash"),
        status,
        started,
        delta,
        stopped,
        _completed(model="gemini-3.8-flash"),
        model="gemini-3.8-flash",
    )

    assert [event.type for event in result] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event.sequence_number for event in result] == list(range(10))
    assert isinstance(result[0], ResponseCreatedEvent)
    assert result[0].response.model == "gemini-3.8-flash"
    assert isinstance(result[1], ResponseInProgressEvent)
    assert isinstance(result[2], ResponseOutputItemAddedEvent)
    assert isinstance(result[2].item, ResponseOutputMessage)
    assert result[2].item.content == []
    assert isinstance(result[3], ResponseContentPartAddedEvent)
    assert isinstance(result[3].part, ResponseOutputText)
    assert result[3].part.text == ""
    assert isinstance(result[4], ResponseTextDeltaEvent)
    assert result[4].delta == "Hello"
    assert isinstance(result[5], ResponseTextDeltaEvent)
    assert result[5].delta == " world"
    assert isinstance(result[6], ResponseTextDoneEvent)
    assert result[6].text == "Hello world"
    assert isinstance(result[7], ResponseContentPartDoneEvent)
    assert isinstance(result[7].part, ResponseOutputText)
    assert result[7].part.text == "Hello world"
    assert isinstance(result[8], ResponseOutputItemDoneEvent)
    assert isinstance(result[8].item, ResponseOutputMessage)
    assert isinstance(result[8].item.content[0], ResponseOutputText)
    assert result[8].item.content[0].text == "Hello world"
    terminal = result[9]
    assert isinstance(terminal, ResponseCompletedEvent)
    assert terminal.response.output_text == "Hello world"


@pytest.mark.asyncio
async def test_convert_interaction_stream_keeps_output_indices_contiguous() -> None:
    user_started = StepStart(index=0, step=UserInputStep())
    model_started = StepStart(index=1, step=ModelOutputStep())

    result = await _converted_events(
        _created(),
        user_started,
        StepStop(index=0),
        model_started,
        StepDelta(index=1, delta=TextDelta(text="Hello")),
        StepStop(index=1),
        _completed(),
    )

    added = next(event for event in result if isinstance(event, ResponseOutputItemAddedEvent))
    assert added.output_index == 0
    assert added.item.id == "msg-0"
    terminal = result[-1]
    assert isinstance(terminal, ResponseCompletedEvent)
    assert terminal.response.output[0].id == "msg-0"


@pytest.mark.asyncio
async def test_convert_interaction_stream_uses_terminal_steps_when_present() -> None:
    result = await _converted_events(
        _created(),
        _completed(steps=[ModelOutputStep(content=[TextContent(text="terminal")])]),
    )

    terminal = result[-1]
    assert isinstance(terminal, ResponseCompletedEvent)
    assert terminal.response.model == "requested"
    assert terminal.response.output_text == "terminal"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "event_type", "message_status"),
    [
        ("failed", "response.failed", "incomplete"),
        ("incomplete", "response.incomplete", "incomplete"),
    ],
)
async def test_convert_interaction_stream_maps_non_success_terminal_status(
    status: InteractionSseEventInteractionStatus,
    event_type: str,
    message_status: str,
) -> None:
    result = await _converted_events(
        _created(),
        _completed(status, steps=[ModelOutputStep(content=[TextContent(text="partial")])]),
    )

    terminal = result[-1]
    assert isinstance(terminal, ResponseFailedEvent | ResponseIncompleteEvent)
    assert terminal.type == event_type
    assert isinstance(terminal.response.output[0], ResponseOutputMessage)
    assert terminal.response.output[0].status == message_status


@pytest.mark.asyncio
async def test_convert_interaction_stream_logs_and_skips_unknown_event(caplog: pytest.LogCaptureFixture) -> None:
    unknown = UnknownInteractionSSEEvent(raw={"event_type": "future.event", "value": 1})

    with caplog.at_level(logging.WARNING, logger="any_llm"):
        result = await _converted_events(_created(), unknown, _completed())

    assert [event.type for event in result] == [
        "response.created",
        "response.in_progress",
        "response.completed",
    ]
    assert "Skipping unknown Gemini Interactions event" in caplog.text


@pytest.mark.asyncio
async def test_convert_interaction_stream_logs_and_skips_unknown_delta(caplog: pytest.LogCaptureFixture) -> None:
    started = StepStart(index=0, step=ModelOutputStep())
    non_text = StepDelta(index=0, delta=ArgumentsDelta(arguments="{}"))
    unknown = StepDelta(index=0, delta=UnknownStepDeltaData(raw={"type": "future_delta", "value": 1}))
    stopped = StepStop(index=0)

    with caplog.at_level(logging.WARNING, logger="any_llm"):
        result = await _converted_events(_created(), started, non_text, unknown, stopped, _completed())

    assert [event.type for event in result] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert "Skipping unknown Gemini Interactions step delta" in caplog.text


@pytest.mark.asyncio
async def test_convert_interaction_stream_raises_error_event() -> None:
    error = ErrorEvent.model_validate(
        {"event_type": "error", "error": {"code": "gateway_timeout", "message": "deadline expired"}}
    )

    with pytest.raises(ProviderError, match="deadline expired") as raised:
        _ = [event async for event in convert_interaction_stream(_events(error), model="requested")]

    assert raised.value.code == "gateway_timeout"


@pytest.mark.asyncio
async def test_convert_interaction_stream_rejects_missing_terminal_event() -> None:
    with pytest.raises(ProviderError, match=r"before interaction\.completed"):
        await _converted_events(_created())


@pytest.mark.asyncio
async def test_convert_interaction_stream_rejects_terminal_before_created() -> None:
    with pytest.raises(ProviderError, match=r"before interaction\.created"):
        await _converted_events(_completed("failed"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "message"),
    [
        (
            [StepStart(index=0, step=ModelOutputStep())],
            "step.start before interaction.created",
        ),
        (
            [InteractionStatusUpdate(interaction_id="int-123", status="in_progress")],
            "status update before interaction.created",
        ),
        (
            [StepDelta(index=0, delta=TextDelta(text="unexpected"))],
            "step.delta before interaction.created",
        ),
        (
            [StepStop(index=0)],
            "step.stop before interaction.created",
        ),
        (
            [_created(), _created()],
            "interaction.created more than once",
        ),
        (
            [
                _created(),
                StepStart(index=0, step=ModelOutputStep()),
                StepStart(index=0, step=ModelOutputStep()),
            ],
            "started step 0 more than once",
        ),
        (
            [
                _created(),
                StepDelta(index=0, delta=TextDelta(text="unexpected")),
            ],
            "delta before step.start",
        ),
        (
            [
                _created(),
                StepStart(index=0, step=UserInputStep()),
                StepDelta(index=0, delta=TextDelta(text="unexpected")),
            ],
            "text for non-model step",
        ),
        (
            [_created(), StepStop(index=0)],
            "stopped unknown step",
        ),
        (
            [
                _created(),
                StepStart(index=0, step=ModelOutputStep()),
                _completed(),
            ],
            "before step.stop",
        ),
    ],
)
async def test_convert_interaction_stream_rejects_malformed_order(
    events: list[InteractionSSEEvent],
    message: str,
) -> None:
    with pytest.raises(ProviderError, match=message):
        await _converted_events(*events)


@pytest.mark.asyncio
async def test_convert_interaction_stream_closes_source_when_consumer_stops() -> None:
    stream = AsyncMock()
    stream.__aiter__.return_value = [_created()]

    converted = convert_interaction_stream(stream, model="requested")
    await anext(converted)
    await converted.aclose()

    stream.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_convert_interaction_stream_propagates_close_error_after_success() -> None:
    stream = AsyncMock()
    stream.__aiter__.return_value = [_created(), _completed()]
    stream.close.side_effect = RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        _ = [event async for event in convert_interaction_stream(stream, model="requested")]


@pytest.mark.asyncio
async def test_convert_interaction_stream_preserves_primary_error_when_close_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = AsyncMock()
    stream.__aiter__.return_value = [
        ErrorEvent.model_validate(
            {"event_type": "error", "error": {"code": "gateway_timeout", "message": "request failed"}}
        )
    ]
    stream.close.side_effect = RuntimeError("close failed")

    with (
        caplog.at_level(logging.WARNING, logger="any_llm"),
        pytest.raises(ProviderError, match="request failed"),
    ):
        _ = [event async for event in convert_interaction_stream(stream, model="requested")]

    assert "Failed to close Gemini Interactions stream" in caplog.text


@pytest.mark.asyncio
async def test_convert_interaction_stream_propagates_cancellation_and_closes_source() -> None:
    stream = AsyncMock()

    async def blocked_events() -> AsyncIterator[InteractionSSEEvent]:
        yield _created()
        await asyncio.Event().wait()

    stream.__aiter__.side_effect = blocked_events
    converted = convert_interaction_stream(stream, model="requested")
    await anext(converted)
    await anext(converted)
    pending = asyncio.create_task(anext(converted))
    await asyncio.sleep(0)
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    stream.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_aresponses_defaults_only_interactions_requests_to_v1() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        client.aio.interactions.create = AsyncMock(return_value=_interaction())
        provider = GeminiProvider(api_key="test-key")
        result = await provider.aresponses(
            "gemini-3.8-flash",
            "Hello",
            instructions="Be concise",
            timeout=1.5,
        )

    assert isinstance(result, Response)
    assert result.output_text == "Hello"
    client_class.assert_called_once_with(api_key="test-key")
    client.aio.interactions.create.assert_awaited_once_with(
        api_version="v1",
        model="gemini-3.8-flash",
        input="Hello",
        system_instruction="Be concise",
        timeout=1.5,
    )


@pytest.mark.asyncio
async def test_aresponses_preserves_explicit_v1beta_client_configuration() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        client.aio.interactions.create = AsyncMock(return_value=_interaction())
        provider = GeminiProvider(api_key="test-key", http_options={"api_version": "v1beta"})
        await provider.aresponses("gemini-3.8-flash", "Hello")

    assert client_class.call_args.kwargs["http_options"] == {"api_version": "v1beta"}
    assert client.aio.interactions.create.await_args.kwargs["api_version"] == "v1beta"


@pytest.mark.asyncio
async def test_aresponses_rejects_openai_extra_body() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client"):
        provider = GeminiProvider(api_key="test-key")
        with pytest.raises(UnsupportedParameterError, match="extra_body"):
            await provider.aresponses(
                "gemini-3.8-flash",
                "Hello",
                extra_body={"future": True},
            )
