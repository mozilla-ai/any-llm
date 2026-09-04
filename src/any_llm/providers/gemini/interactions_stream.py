from __future__ import annotations

from inspect import isawaitable
from typing import TYPE_CHECKING, NoReturn, Protocol, assert_never, runtime_checkable

from google.genai.interactions import (
    ErrorEvent,
    InteractionCompletedEvent,
    InteractionCreatedEvent,
    InteractionSSEEvent,
    InteractionStatusUpdate,
    ModelOutputStep,
    StepDelta,
    StepStart,
    StepStop,
    TextContent,
    TextDelta,
    UnknownInteractionSSEEvent,
    UnknownStepDeltaData,
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

from any_llm.exceptions import ProviderError
from any_llm.logging import logger

from .interactions import _response_from_interaction

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable

    from openai.types.responses import Response as OpenAIResponse

    from any_llm.types.responses import ResponseStreamEvent


@runtime_checkable
class _Closeable(Protocol):
    def close(self) -> Awaitable[None] | None: ...


def _terminal_event(response: OpenAIResponse, sequence_number: int) -> ResponseStreamEvent:
    if response.status == "completed":
        return ResponseCompletedEvent(
            type="response.completed",
            sequence_number=sequence_number,
            response=response,
        )
    if response.status == "failed":
        return ResponseFailedEvent(
            type="response.failed",
            sequence_number=sequence_number,
            response=response,
        )
    return ResponseIncompleteEvent(
        type="response.incomplete",
        sequence_number=sequence_number,
        response=response,
    )


def _raise_stream_error(message: str, *, code: str | None = None) -> NoReturn:
    raise ProviderError(message, provider_name="gemini", code=code)


class _TextStreamState:
    """Track Gemini steps while emitting the OpenAI text event lifecycle.

    Gemini streams step.start, step.delta, and step.stop. OpenAI consumers
    expect item and content-part boundaries around text deltas, so this state
    stays in the provider adapter instead of leaking Gemini events to callers.
    https://ai.google.dev/gemini-api/docs/interactions/streaming
    https://platform.openai.com/docs/api-reference/responses-streaming
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.sequence = 0
        self.started = False
        self.open_steps: set[int] = set()
        self.text_by_step: dict[int, str] = {}
        self.output_index_by_step: dict[int, int] = {}

    def convert(self, event: InteractionSSEEvent) -> tuple[list[ResponseStreamEvent], bool]:
        terminal = False
        if isinstance(event, InteractionCreatedEvent):
            converted = self._created(event)
        elif isinstance(event, StepStart):
            converted = self._step_started(event)
        elif isinstance(event, StepDelta):
            converted = self._step_delta(event)
        elif isinstance(event, StepStop):
            converted = self._step_stopped(event)
        elif isinstance(event, InteractionCompletedEvent):
            converted = [self._completed(event)]
            terminal = True
        elif isinstance(event, ErrorEvent):
            self._error(event)
        elif isinstance(event, UnknownInteractionSSEEvent):
            logger.warning("Skipping unknown Gemini Interactions event: %s", event.event_type)
            converted = []
        elif isinstance(event, InteractionStatusUpdate):
            if not self.started:
                _raise_stream_error("Gemini interaction stream emitted a status update before interaction.created")
            converted = []
        else:
            assert_never(event)
        return converted, terminal

    def incomplete(self) -> NoReturn:
        _raise_stream_error("Gemini interaction stream ended before interaction.completed")

    def _next_sequence(self) -> int:
        sequence = self.sequence
        self.sequence += 1
        return sequence

    def _created(self, event: InteractionCreatedEvent) -> list[ResponseStreamEvent]:
        if self.started:
            _raise_stream_error("Gemini interaction stream emitted interaction.created more than once")
        self.started = True
        response = _response_from_interaction(event.interaction, fallback_model=self.model)
        return [
            ResponseCreatedEvent(
                type="response.created",
                sequence_number=self._next_sequence(),
                response=response,
            ),
            ResponseInProgressEvent(
                type="response.in_progress",
                sequence_number=self._next_sequence(),
                response=response.model_copy(update={"status": "in_progress"}),
            ),
        ]

    def _step_started(self, event: StepStart) -> list[ResponseStreamEvent]:
        if not self.started:
            _raise_stream_error("Gemini interaction stream emitted step.start before interaction.created")
        if event.index in self.open_steps or event.index in self.text_by_step:
            _raise_stream_error(f"Gemini interaction stream started step {event.index} more than once")
        self.open_steps.add(event.index)
        if not isinstance(event.step, ModelOutputStep):
            return []

        prefix = "".join(part.text for part in event.step.content or [] if isinstance(part, TextContent))
        self.text_by_step[event.index] = prefix
        output_index = len(self.output_index_by_step)
        self.output_index_by_step[event.index] = output_index
        item_id = f"msg-{output_index}"
        events: list[ResponseStreamEvent] = [
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                sequence_number=self._next_sequence(),
                output_index=output_index,
                item=ResponseOutputMessage(
                    id=item_id,
                    type="message",
                    role="assistant",
                    status="in_progress",
                    content=[],
                ),
            ),
            ResponseContentPartAddedEvent(
                type="response.content_part.added",
                sequence_number=self._next_sequence(),
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                part=ResponseOutputText(type="output_text", text="", annotations=[]),
            ),
        ]
        if prefix:
            events.append(self._text_delta(event.index, prefix))
        return events

    def _step_delta(self, event: StepDelta) -> list[ResponseStreamEvent]:
        if not self.started:
            _raise_stream_error("Gemini interaction stream emitted step.delta before interaction.created")
        if event.index not in self.open_steps:
            _raise_stream_error(f"Gemini interaction stream emitted a delta before step.start for step {event.index}")
        if isinstance(event.delta, UnknownStepDeltaData):
            logger.warning("Skipping unknown Gemini Interactions step delta: %s", event.delta.raw)
            return []
        if not isinstance(event.delta, TextDelta):
            return []
        if event.index not in self.text_by_step:
            _raise_stream_error(f"Gemini interaction stream emitted text for non-model step {event.index}")
        self.text_by_step[event.index] += event.delta.text
        return [self._text_delta(event.index, event.delta.text)]

    def _text_delta(self, step_index: int, text: str) -> ResponseTextDeltaEvent:
        output_index = self.output_index_by_step[step_index]
        return ResponseTextDeltaEvent(
            type="response.output_text.delta",
            sequence_number=self._next_sequence(),
            item_id=f"msg-{output_index}",
            output_index=output_index,
            content_index=0,
            delta=text,
            logprobs=[],
        )

    def _step_stopped(self, event: StepStop) -> list[ResponseStreamEvent]:
        if not self.started:
            _raise_stream_error("Gemini interaction stream emitted step.stop before interaction.created")
        if event.index not in self.open_steps:
            _raise_stream_error(f"Gemini interaction stream stopped unknown step {event.index}")
        self.open_steps.remove(event.index)
        if event.index not in self.text_by_step:
            return []

        text = self.text_by_step[event.index]
        output_index = self.output_index_by_step[event.index]
        item_id = f"msg-{output_index}"
        completed_part = ResponseOutputText(type="output_text", text=text, annotations=[])
        completed_item = ResponseOutputMessage(
            id=item_id,
            type="message",
            role="assistant",
            status="completed",
            content=[completed_part],
        )
        return [
            ResponseTextDoneEvent(
                type="response.output_text.done",
                sequence_number=self._next_sequence(),
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                text=text,
                logprobs=[],
            ),
            ResponseContentPartDoneEvent(
                type="response.content_part.done",
                sequence_number=self._next_sequence(),
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                part=completed_part,
            ),
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                sequence_number=self._next_sequence(),
                output_index=output_index,
                item=completed_item,
            ),
        ]

    def _completed(self, event: InteractionCompletedEvent) -> ResponseStreamEvent:
        if not self.started:
            _raise_stream_error("Gemini interaction stream completed before interaction.created")
        if self.open_steps:
            _raise_stream_error(f"Gemini interaction stream completed before step.stop for step {min(self.open_steps)}")
        response = _response_from_interaction(event.interaction, fallback_model=self.model)
        if not response.output:
            response = response.model_copy(update={"output": self._stream_messages()})
        return _terminal_event(response, self._next_sequence())

    def _stream_messages(self) -> list[ResponseOutputMessage]:
        return [
            ResponseOutputMessage(
                id=f"msg-{self.output_index_by_step[step_index]}",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
            )
            for step_index, text in sorted(
                self.text_by_step.items(),
                key=lambda item: self.output_index_by_step[item[0]],
            )
        ]

    @staticmethod
    def _error(event: ErrorEvent) -> NoReturn:
        message = (
            event.error.message if event.error is not None and event.error.message else "Gemini interaction failed"
        )
        code = event.error.code if event.error is not None else None
        _raise_stream_error(message, code=code)


async def convert_interaction_stream(
    stream: AsyncIterator[InteractionSSEEvent],
    *,
    model: str,
) -> AsyncGenerator[ResponseStreamEvent]:
    """Normalize a create stream into the OpenAI text event lifecycle."""
    state = _TextStreamState(model)
    primary_error: BaseException | None = None
    try:
        async for event in stream:
            events, terminal = state.convert(event)
            for converted_event in events:
                yield converted_event
            if terminal:
                return
        state.incomplete()
    except BaseException as error:
        if not isinstance(error, GeneratorExit):
            primary_error = error
        raise
    finally:
        try:
            if isinstance(stream, _Closeable) and isawaitable(close_result := stream.close()):
                await close_result
        except BaseException as close_error:
            if primary_error is None:
                raise
            logger.warning(
                "Failed to close Gemini Interactions stream while handling another error",
                exc_info=close_error,
            )
