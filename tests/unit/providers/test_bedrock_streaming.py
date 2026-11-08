import asyncio
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any, cast
from unittest.mock import MagicMock, Mock

import pytest
from botocore.exceptions import EventStreamError

from any_llm.exceptions import RateLimitError
from any_llm.providers.bedrock import BedrockProvider
from any_llm.types.completion import ChatCompletionChunk

MODEL = "amazon.nova-lite-v1:0"


class _FakeEventStream:
    """Stands in for botocore's ``EventStream``: iterable once, and closable."""

    def __init__(self, events: list[dict[str, Any]], error: Exception | None = None) -> None:
        self._events = events
        self._error = error
        self.read_threads: list[int] = []
        self.closed = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for event in self._events:
            self.read_threads.append(threading.get_ident())
            yield event
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.closed = True


def _text(text: str) -> dict[str, Any]:
    return {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}}


async def _open_stream(event_stream: _FakeEventStream, **kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
    client = Mock()
    client.converse_stream.return_value = {"stream": event_stream}
    provider = BedrockProvider(client=client, unified_exceptions=True)
    stream = await provider.acompletion(
        model=MODEL, messages=[{"role": "user", "content": "Hello"}], stream=True, **kwargs
    )
    assert isinstance(stream, AsyncIterator)
    return stream


async def _next_chunk(stream: AsyncIterator[ChatCompletionChunk]) -> ChatCompletionChunk:
    return await anext(stream)


@pytest.mark.asyncio
async def test_converse_stream_reads_each_event_off_the_event_loop_on_demand() -> None:
    event_stream = _FakeEventStream([_text("first"), _text("second")])
    stream = await _open_stream(event_stream)

    assert event_stream.read_threads == []
    first = await anext(stream)
    assert first.choices[0].delta.content == "first"
    assert len(event_stream.read_threads) == 1
    assert event_stream.read_threads[0] != threading.get_ident()

    rest = [chunk async for chunk in stream]
    assert [chunk.choices[0].delta.content for chunk in rest] == ["second"]
    assert event_stream.closed


@pytest.mark.asyncio
async def test_converse_stream_keeps_tool_indices_and_skips_unknown_events() -> None:
    event_stream = _FakeEventStream(
        [
            {"unknownEvent": {}},
            _text("hello"),
            {"contentBlockStart": {"contentBlockIndex": 3, "start": {"toolUse": {"toolUseId": "a", "name": "one"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 3, "delta": {"toolUse": {"input": "{}"}}}},
            {"contentBlockStart": {"contentBlockIndex": 7, "start": {"toolUse": {"toolUseId": "b", "name": "two"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 7, "delta": {"toolUse": {"input": '{"x":1}'}}}},
        ]
    )
    stream = await _open_stream(event_stream)
    chunks = [chunk async for chunk in stream]

    assert chunks[0].choices[0].delta.content == "hello"
    tool_calls = [chunk.choices[0].delta.tool_calls for chunk in chunks[1:]]
    assert [calls[0].index for calls in tool_calls if calls] == [0, 0, 1, 1]
    assert event_stream.closed


@pytest.mark.asyncio
async def test_converse_stream_error_is_converted_and_closes_the_stream() -> None:
    error = EventStreamError(  # type: ignore[no-untyped-call]
        {"Error": {"Code": "ThrottlingException", "Message": "Too many requests"}}, "ConverseStream"
    )
    event_stream = _FakeEventStream([_text("first")], error=error)
    stream = await _open_stream(event_stream)

    await anext(stream)
    with pytest.raises(RateLimitError) as raised:
        await anext(stream)

    assert raised.value.original_exception is error
    assert event_stream.closed


@pytest.mark.asyncio
async def test_converse_stream_closed_early_closes_the_bedrock_stream() -> None:
    event_stream = _FakeEventStream([_text("first"), _text("second")])

    stream = cast("AsyncGenerator[ChatCompletionChunk]", BedrockProvider._stream_converse(event_stream, MODEL))
    await anext(stream)
    await stream.aclose()

    assert len(event_stream.read_threads) == 1
    assert event_stream.closed


@pytest.mark.asyncio
async def test_converse_stream_does_not_block_the_event_loop() -> None:
    release = threading.Event()
    reading = threading.Event()

    def blocking_events() -> Iterator[dict[str, Any]]:
        reading.set()
        assert release.wait(5)
        yield _text("late")

    event_stream = MagicMock()
    event_stream.__iter__.return_value = blocking_events()
    stream = BedrockProvider._stream_converse(event_stream, MODEL)
    read = asyncio.create_task(_next_chunk(stream))

    while not reading.is_set():  # noqa: ASYNC110
        await asyncio.sleep(0.001)
    assert not read.done()
    release.set()

    assert (await read).choices[0].delta.content == "late"


def test_converse_sync_stream_closes_the_bedrock_stream() -> None:
    event_stream = _FakeEventStream([_text("first"), _text("second")])
    client = Mock()
    client.converse_stream.return_value = {"stream": event_stream}

    stream = BedrockProvider(client=client).completion(
        model=MODEL, messages=[{"role": "user", "content": "Hello"}], stream=True
    )
    chunks = list(stream)

    assert [chunk.choices[0].delta.content for chunk in chunks] == ["first", "second"]
    assert event_stream.closed


_USAGE_EVENT: dict[str, Any] = {"metadata": {"usage": {"inputTokens": 33, "outputTokens": 20, "totalTokens": 53}}}


@pytest.mark.asyncio
@pytest.mark.parametrize(("stream_options", "choices"), [({"include_usage": True}, 0), (None, 1)])
async def test_converse_stream_usage_chunk_follows_include_usage(
    stream_options: dict[str, Any] | None, choices: int
) -> None:
    """The stream's last chunk carries usage; it has no choices only when usage was requested."""
    events = [_text("Hi"), {"messageStop": {"stopReason": "end_turn"}}, _USAGE_EVENT]
    stream = await _open_stream(_FakeEventStream(events), stream_options=stream_options)

    chunks = [chunk async for chunk in stream]

    last = chunks[-1]
    assert len(last.choices) == choices
    assert last.usage is not None
    assert (last.usage.prompt_tokens, last.usage.completion_tokens) == (33, 20)
