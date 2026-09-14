import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from google.genai import types
from google.genai._gaos.types.interactions import Interaction
from google.genai.interactions import (
    ArgumentsDelta,
    Error,
    ErrorEvent,
    ImageContent,
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
    ThoughtStep,
    UnknownInteractionSSEEvent,
    UnknownStep,
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
    created: str | None = "2026-01-02T03:04:05Z",
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
    assert message.id == "msg-int-123-0"
    assert response.usage is not None
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 6
    assert response.usage.input_tokens_details.cached_tokens == 1
    assert response.usage.output_tokens_details.reasoning_tokens == 3


def test_convert_interaction_preserves_explicit_zero_total_usage() -> None:
    usage = Usage(total_input_tokens=2, total_output_tokens=3, total_tokens=0)
    response = convert_interaction_to_response(_interaction(usage=usage))

    assert response.usage is not None
    assert response.usage.total_tokens == 0


def test_convert_interaction_preserves_absent_usage() -> None:
    interaction = _interaction()
    interaction.usage = None

    assert convert_interaction_to_response(interaction).usage is None


def test_convert_interaction_preserves_empty_text_output() -> None:
    response = convert_interaction_to_response(_interaction(steps=[ModelOutputStep(content=[TextContent(text="")])]))

    assert len(response.output) == 1
    assert response.output_text == ""


def test_convert_interaction_ignores_non_output_steps() -> None:
    interaction = _interaction(
        steps=[
            {"type": "future_step", "future": True},
            ModelOutputStep(content=[TextContent(text="kept")]),
        ]
    )

    response = convert_interaction_to_response(interaction)

    assert response.output_text == "kept"
    assert len(response.output) == 1
    assert response.output[0].id == "msg-int-123-0"


@pytest.mark.parametrize(
    "content",
    [
        [ImageContent(data="aW1hZ2U=", mime_type="image/png")],
        [TextContent(text="partial"), ImageContent(data="aW1hZ2U=", mime_type="image/png")],
        [TextContent(text="partial"), {"type": "future_content", "future": True}],
    ],
)
def test_convert_interaction_rejects_non_text_model_output(content: list[object]) -> None:
    step = ModelOutputStep.model_validate({"content": content})

    with pytest.raises(ProviderError, match="non-text model output"):
        convert_interaction_to_response(_interaction(steps=[step]))


def test_convert_interaction_keeps_text_after_thought_output() -> None:
    response = convert_interaction_to_response(
        _interaction(steps=[ThoughtStep(signature="opaque"), ModelOutputStep(content=[TextContent(text="25")])])
    )

    assert response.output_text == "25"
    assert len(response.output) == 1
    assert response.output[0].id == "msg-int-123-0"


@pytest.mark.parametrize("identifiers", [("first", "second"), (None, None)])
def test_convert_interaction_message_ids_are_unique(identifiers: tuple[str | None, str | None]) -> None:
    responses = [
        convert_interaction_to_response(
            _interaction(
                steps=[
                    ModelOutputStep(content=[TextContent(text="first part")]),
                    ModelOutputStep(content=[TextContent(text="second part")]),
                ]
            ).model_copy(update={"id": identifier})
        )
        for identifier in identifiers
    ]

    assert len({item.id for response in responses for item in response.output}) == 4
    for response in responses:
        assert [item.id for item in response.output] == [f"msg-{response.id}-0", f"msg-{response.id}-1"]


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (Error(code="gateway_timeout", message="deadline expired"), "deadline expired"),
        (Error(code="gateway_timeout"), "gateway_timeout"),
        (Error(), "Gemini interaction failed"),
    ],
)
def test_convert_interaction_maps_provider_error_without_raw_side_channel(error: Error, expected_message: str) -> None:
    interaction = _interaction(status="failed", steps=[])
    interaction.errors = [error]

    response = convert_interaction_to_response(interaction)

    assert response.error is not None
    assert response.error.code == "server_error"
    assert response.error.message == expected_message
    assert not any(name.startswith("gemini_") for name in response.model_dump())


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="Process timezone control requires time.tzset")
@pytest.mark.parametrize("timezone", ["UTC0", "EST5"])
@pytest.mark.parametrize(
    "created",
    ["2026-01-02T03:04:05", "2026-01-02T03:04:05Z", "2026-01-02T08:34:05+05:30"],
)
def test_convert_interaction_timestamp_is_independent_of_process_timezone(
    monkeypatch: pytest.MonkeyPatch, timezone: str, created: str
) -> None:
    try:
        with monkeypatch.context() as environment:
            environment.setenv("TZ", timezone)
            time.tzset()
            response = convert_interaction_to_response(_interaction(created=created))
    finally:
        time.tzset()

    assert response.created_at == 1767323045.0


@pytest.mark.parametrize("created", ["invalid", None, ""])
def test_convert_interaction_handles_unknown_status_and_invalid_timestamp(created: str | None) -> None:
    response = convert_interaction_to_response(_interaction(status="future_status", created=created))

    assert response.status == "in_progress"
    assert response.created_at == 0.0


@pytest.mark.parametrize(
    ("gemini_status", "expected_status"),
    [
        ("queued", "queued"),
        ("requires_action", "incomplete"),
        ("budget_exceeded", "incomplete"),
    ],
)
def test_convert_interaction_normalizes_extended_sdk_statuses(
    gemini_status: str,
    expected_status: str,
) -> None:
    assert convert_interaction_to_response(_interaction(status=gemini_status)).status == expected_status


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
    assert added.item.id == "msg-int-123-0"
    terminal = result[-1]
    assert isinstance(terminal, ResponseCompletedEvent)
    assert terminal.response.output[0].id == "msg-int-123-0"


@pytest.mark.asyncio
async def test_convert_interaction_stream_requires_non_model_steps_to_stop() -> None:
    with pytest.raises(ProviderError, match=r"before step\.stop for step 0"):
        await _converted_events(
            _created(),
            StepStart(index=0, step=UserInputStep()),
            _completed(),
        )


@pytest.mark.asyncio
async def test_convert_interaction_stream_orders_terminal_messages_by_output_index() -> None:
    result = await _converted_events(
        _created(),
        StepStart(index=7, step=ModelOutputStep()),
        StepStart(index=2, step=ModelOutputStep()),
        StepDelta(index=2, delta=TextDelta(text="second")),
        StepStop(index=2),
        StepDelta(index=7, delta=TextDelta(text="first")),
        StepStop(index=7),
        _completed(),
    )

    terminal = result[-1]
    assert isinstance(terminal, ResponseCompletedEvent)
    assert [message.id for message in terminal.response.output] == ["msg-int-123-0", "msg-int-123-1"]
    assert terminal.response.output_text == "firstsecond"


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
@pytest.mark.parametrize("status", ["completed", "incomplete", "failed", "cancelled"])
@pytest.mark.parametrize("terminal_id", ["int-123", None])
async def test_convert_interaction_stream_normalizes_equivalent_terminal_snapshots(
    status: InteractionSseEventInteractionStatus,
    terminal_id: str | None,
) -> None:
    streamed_steps: list[InteractionSSEEvent] = [
        StepStart(index=7, step=ModelOutputStep(content=[TextContent(text="first")])),
        StepDelta(index=7, delta=TextDelta(text=" step")),
        StepStop(index=7),
        StepStart(index=2, step=ModelOutputStep(content=[TextContent(text="")])),
        StepStop(index=2),
        StepStart(index=9, step=ModelOutputStep()),
        StepDelta(index=9, delta=TextDelta(text="third")),
        StepStop(index=9),
    ]
    terminal_steps: list[Step] = [
        ModelOutputStep(content=[TextContent(text="first step")]),
        ModelOutputStep(content=[TextContent(text="")]),
        ModelOutputStep(content=[TextContent(text="third")]),
    ]

    terminal_event = _completed(status)
    terminal_event.interaction = terminal_event.interaction.model_copy(update={"id": terminal_id})
    without_snapshot = await _converted_events(_created(), *streamed_steps, terminal_event)
    with_snapshot = await _converted_events(_created(), *streamed_steps, _completed(status, steps=terminal_steps))

    assert without_snapshot[-1].model_dump(mode="json") == with_snapshot[-1].model_dump(mode="json")
    terminal = without_snapshot[-1]
    assert isinstance(terminal, ResponseCompletedEvent | ResponseFailedEvent | ResponseIncompleteEvent)
    assert terminal.response.id == "int-123"
    assert terminal.response.status == status
    done_items = [event.item for event in without_snapshot if isinstance(event, ResponseOutputItemDoneEvent)]
    assert terminal.response.output == done_items
    for item in done_items:
        assert isinstance(item, ResponseOutputMessage)
        assert item.status == "completed"
    assert [item.id for item in done_items] == ["msg-int-123-0", "msg-int-123-1", "msg-int-123-2"]
    for event in without_snapshot:
        if isinstance(event, ResponseTextDeltaEvent):
            assert event.item_id == done_items[event.output_index].id


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
async def test_convert_interaction_stream_logs_and_skips_unknown_delta(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prefix: list[InteractionSSEEvent] = [
        _created(),
        StepStart(index=7, step=ModelOutputStep(content=[TextContent(text="A")])),
    ]
    suffix: list[InteractionSSEEvent] = [StepDelta(index=7, delta=TextDelta(text="Z")), StepStop(index=7), _completed()]
    unknown = StepDelta(index=7, delta=UnknownStepDeltaData(raw={"type": "future_delta", "value": 1}))
    expected = await _converted_events(*prefix, *suffix)

    with caplog.at_level(logging.WARNING, logger="any_llm"):
        actual = await _converted_events(*prefix, unknown, *suffix)

    assert [event.model_dump() for event in actual] == [event.model_dump() for event in expected]
    assert isinstance(actual[-1], ResponseCompletedEvent)
    assert actual[-1].response.output_text == "AZ"
    assert "Skipping unknown Gemini Interactions delta" in caplog.text


@pytest.mark.asyncio
async def test_convert_interaction_stream_rejects_unsupported_model_delta() -> None:
    with pytest.raises(ProviderError, match="non-text model output delta"):
        await _converted_events(
            _created(),
            StepStart(index=0, step=ModelOutputStep()),
            StepDelta(index=0, delta=ArgumentsDelta(arguments="{}")),
        )


@pytest.mark.asyncio
async def test_convert_interaction_stream_ignores_non_model_step_deltas() -> None:
    result = await _converted_events(
        _created(),
        StepStart(index=0, step=UserInputStep()),
        StepDelta(index=0, delta=ArgumentsDelta(arguments="{}")),
        StepStop(index=0),
        _completed(),
    )

    assert [event.type for event in result] == ["response.created", "response.in_progress", "response.completed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "step",
    [
        ModelOutputStep(content=[ImageContent(data="aW1hZ2U=", mime_type="image/png")]),
        ModelOutputStep(content=[TextContent(text="partial"), ImageContent(data="aW1hZ2U=", mime_type="image/png")]),
        UnknownStep(raw={"type": "future_output"}),
    ],
)
async def test_convert_interaction_stream_rejects_unsupported_output_and_closes_source(step: Step) -> None:
    stream = AsyncMock()
    stream.close = AsyncMock()
    stream.__aiter__.return_value = [_created(), StepStart(index=0, step=step)]

    with pytest.raises(ProviderError, match="model output"):
        _ = [event async for event in convert_interaction_stream(stream, model="requested")]

    stream.close.assert_awaited_once_with()


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
    stream.close = AsyncMock()
    stream.__aiter__.return_value = [_created()]

    converted = convert_interaction_stream(stream, model="requested")
    await anext(converted)
    await converted.aclose()

    stream.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_convert_interaction_stream_propagates_close_error_after_success() -> None:
    stream = AsyncMock()
    stream.close = AsyncMock(side_effect=RuntimeError("close failed"))
    stream.__aiter__.return_value = [_created(), _completed()]

    with pytest.raises(RuntimeError, match="close failed"):
        _ = [event async for event in convert_interaction_stream(stream, model="requested")]


@pytest.mark.asyncio
async def test_convert_interaction_stream_preserves_primary_error_when_close_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = AsyncMock()
    stream.close = AsyncMock(side_effect=RuntimeError("close failed"))
    stream.__aiter__.return_value = [
        ErrorEvent.model_validate(
            {"event_type": "error", "error": {"code": "gateway_timeout", "message": "request failed"}}
        )
    ]

    with (
        caplog.at_level(logging.WARNING, logger="any_llm"),
        pytest.raises(ProviderError, match="request failed"),
    ):
        _ = [event async for event in convert_interaction_stream(stream, model="requested")]

    assert "Failed to close Gemini Interactions stream" in caplog.text


@pytest.mark.asyncio
async def test_convert_interaction_stream_preserves_unsupported_output_error_when_close_fails() -> None:
    stream = AsyncMock()
    stream.close = AsyncMock(side_effect=RuntimeError("close failed"))
    stream.__aiter__.return_value = [
        _created(),
        StepStart(index=0, step=ModelOutputStep(content=[ImageContent(data="aW1hZ2U=", mime_type="image/png")])),
    ]

    with pytest.raises(ProviderError, match="non-text model output"):
        _ = [event async for event in convert_interaction_stream(stream, model="requested")]


@pytest.mark.asyncio
async def test_convert_interaction_stream_propagates_cancellation_and_closes_source() -> None:
    stream = AsyncMock()
    stream.close = AsyncMock()

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


@pytest.mark.asyncio
async def test_aresponses_forwards_sdk_transport_parameters() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        client.aio.interactions.create = AsyncMock(return_value=_interaction())
        provider = GeminiProvider(api_key="test-key")
        await provider._aresponses(
            ResponsesParams(model="gemini-3.8-flash", input="Hello"),
            extra_headers={"x-request-id": "request-123"},
            extra_query={"trace": "enabled"},
        )

    assert client.aio.interactions.create.await_args.kwargs["extra_headers"] == {"x-request-id": "request-123"}
    assert client.aio.interactions.create.await_args.kwargs["extra_query"] == {"trace": "enabled"}


@pytest.mark.asyncio
async def test_aresponses_rejects_unknown_transport_parameter_before_io() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        provider = GeminiProvider(api_key="test-key")

        with pytest.raises(UnsupportedParameterError, match="future_transport"):
            await provider._aresponses(
                ResponsesParams(model="gemini-3.8-flash", input="Hello"),
                future_transport=True,
            )

    client.aio.interactions.create.assert_not_called()


@pytest.mark.asyncio
async def test_aresponses_ignores_none_transport_parameters() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        client.aio.interactions.create = AsyncMock(return_value=_interaction())
        provider = GeminiProvider(api_key="test-key")
        await provider._aresponses(
            ResponsesParams(model="gemini-3.8-flash", input="Hello"),
            extra_headers=None,
            extra_query=None,
            future_transport=None,
        )

    assert "extra_headers" not in client.aio.interactions.create.await_args.kwargs
    assert "extra_query" not in client.aio.interactions.create.await_args.kwargs
    assert "future_transport" not in client.aio.interactions.create.await_args.kwargs


@pytest.mark.asyncio
async def test_private_stream_close_before_iteration_does_not_acquire_source() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        provider = GeminiProvider(api_key="test-key")
        response = await provider._aresponses(ResponsesParams(model="gemini-3.8-flash", input="Hello", stream=True))
        assert isinstance(response, AsyncGenerator)
        await response.aclose()

    client.aio.interactions.create.assert_not_called()


@pytest.mark.asyncio
async def test_public_stream_close_before_iteration_does_not_acquire_source() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        provider = GeminiProvider(api_key="test-key")
        response = await provider.aresponses("gemini-3.8-flash", "Hello", stream=True)
        assert isinstance(response, AsyncGenerator)
        await response.aclose()

    client.aio.interactions.create.assert_not_called()


@pytest.mark.asyncio
async def test_private_stream_close_after_iteration_closes_source() -> None:
    stream = AsyncMock()
    stream.close = AsyncMock()
    stream.__aiter__.side_effect = lambda: _events(_created())
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        client.aio.interactions.create = AsyncMock(return_value=stream)
        provider = GeminiProvider(api_key="test-key")
        response = await provider._aresponses(ResponsesParams(model="gemini-3.8-flash", input="Hello", stream=True))
        assert isinstance(response, AsyncGenerator)
        await anext(response)
        await response.aclose()

    stream.close.assert_awaited_once_with()


@pytest.mark.parametrize("total", [None, 0, 346])
@pytest.mark.asyncio
async def test_real_sdk_serializes_stable_interactions_path_and_body(total: int | None) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "int-123",
                "status": "completed",
                "model": "gemini-3.8-flash",
                "steps": [
                    {"type": "thought", "signature": "opaque"},
                    {"type": "model_output", "content": [{"type": "text", "text": "Hello"}]},
                ],
                "usage": {
                    "total_input_tokens": 11,
                    "total_output_tokens": 90,
                    "total_thought_tokens": 245,
                    "total_tokens": total,
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = GeminiProvider(
            api_key="test-key",
            api_base="https://example.test",
            http_options=types.HttpOptions(httpx_async_client=http_client),
        )
        response = await provider.aresponses(
            "gemini-3.8-flash",
            "Hello",
            instructions="",
            max_output_tokens=0,
        )
    finally:
        await http_client.aclose()

    assert isinstance(response, Response)
    assert response.output_text == "Hello"
    assert response.output[0].id == "msg-int-123-0"
    assert response.usage is not None
    assert response.usage.output_tokens == 335
    assert response.usage.output_tokens_details.reasoning_tokens == 245
    assert response.usage.total_tokens == (346 if total is None else total)
    assert len(requests) == 1
    assert str(requests[0].url) == "https://example.test/v1/interactions"
    assert requests[0].headers["x-goog-api-key"] == "test-key"
    assert json.loads(requests[0].content) == {
        "input": "Hello",
        "model": "gemini-3.8-flash",
        "generation_config": {"max_output_tokens": 0},
        "system_instruction": "",
    }


@pytest.mark.parametrize("total", [None, 0])
@pytest.mark.parametrize("unknown_delta", [False, True])
@pytest.mark.asyncio
async def test_real_sdk_stream_keeps_interleaved_text_after_thought_metadata(
    total: int | None, unknown_delta: bool
) -> None:
    requests: list[httpx.Request] = []
    event_payloads = [
        {
            "event_type": "interaction.created",
            "interaction": {"id": "int-123", "status": "in_progress"},
        },
        {
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "thought"},
        },
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "thought_signature", "signature": "opaque"},
        },
        {"event_type": "step.stop", "index": 0},
        {
            "event_type": "step.start",
            "index": 7,
            "step": {"type": "model_output", "content": [{"type": "text", "text": "A"}]},
        },
        {
            "event_type": "step.start",
            "index": 2,
            "step": {"type": "model_output"},
        },
        {"event_type": "step.delta", "index": 2, "delta": {"type": "text", "text": "bb"}},
        {"event_type": "step.stop", "index": 2},
        {"event_type": "step.delta", "index": 7, "delta": {"type": "text", "text": "C"}},
        {"event_type": "step.stop", "index": 7},
        {
            "event_type": "interaction.completed",
            "interaction": {
                "id": "int-123",
                "status": "completed",
                "usage": {
                    "total_input_tokens": 11,
                    "total_output_tokens": 90,
                    "total_thought_tokens": 245,
                    "total_tokens": total,
                },
            },
        },
    ]
    if unknown_delta:
        event_payloads.insert(5, {"event_type": "step.delta", "index": 7, "delta": {"type": "future_metadata"}})
    body = "".join(f"data: {json.dumps(payload)}\n\n" for payload in event_payloads) + "data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = GeminiProvider(
            api_key="test-key",
            api_base="https://example.test",
            http_options=types.HttpOptions(httpx_async_client=http_client),
        )
        response = await provider.aresponses("gemini-3.8-flash", "Hello", stream=True)
        assert isinstance(response, AsyncIterator)
        events = [event async for event in response]
    finally:
        await http_client.aclose()

    assert [event.sequence_number for event in events] == list(range(len(events)))
    added = [event for event in events if isinstance(event, ResponseOutputItemAddedEvent)]
    assert [event.output_index for event in added] == [0, 1]
    terminal = events[-1]
    assert isinstance(terminal, ResponseCompletedEvent)
    assert [item.id for item in terminal.response.output] == ["msg-int-123-0", "msg-int-123-1"]
    assert terminal.response.output_text == "ACbb"
    assert terminal.response.usage is not None
    assert terminal.response.usage.output_tokens == 335
    assert terminal.response.usage.output_tokens_details.reasoning_tokens == 245
    assert terminal.response.usage.total_tokens == (346 if total is None else total)
    assert str(requests[0].url) == "https://example.test/v1/interactions"
    assert json.loads(requests[0].content)["stream"] is True


@pytest.mark.asyncio
async def test_real_sdk_http_error_uses_unified_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            headers={"content-type": "application/json"},
            json={"error": {"code": "INVALID_ARGUMENT", "message": "invalid input"}},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = GeminiProvider(
            api_key="test-key",
            api_base="https://example.test",
            http_options=types.HttpOptions(httpx_async_client=http_client),
        )
        with pytest.raises(InvalidRequestError) as raised:
            await provider.aresponses("gemini-3.8-flash", "Hello")
    finally:
        await http_client.aclose()

    assert raised.value.status_code == 400
    assert raised.value.code == "INVALID_ARGUMENT"


@pytest.mark.asyncio
async def test_real_sdk_timeout_uses_unified_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

    async def handler(request: httpx.Request) -> httpx.Response:
        message = "read timed out"
        raise httpx.ReadTimeout(message, request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = GeminiProvider(
            api_key="test-key",
            api_base="https://example.test",
            http_options=types.HttpOptions(httpx_async_client=http_client),
        )
        with pytest.raises(ProviderError) as raised:
            await provider.aresponses("gemini-3.8-flash", "Hello", timeout=0.1)
    finally:
        await http_client.aclose()

    assert raised.value.status_code is None
    original = raised.value.original_exception
    assert original is not None
    assert type(original).__name__ == "APITimeoutError"
    assert isinstance(original.__cause__, httpx.ReadTimeout)


def test_responses_calls_interactions_synchronously() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client_class.return_value.aio.interactions.create = AsyncMock(return_value=_interaction())
        provider = GeminiProvider(api_key="test-key")
        response = provider.responses("gemini-3.8-flash", "Hello")

    assert isinstance(response, Response)
    assert response.output_text == "Hello"
