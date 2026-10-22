import asyncio
import threading
from collections.abc import AsyncIterator, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, suppress
from typing import Any, cast
from unittest.mock import Mock

import pytest
from botocore.exceptions import EventStreamError

from any_llm.exceptions import AnyLLMError, RateLimitError
from any_llm.providers.bedrock import BedrockProvider
from any_llm.providers.bedrock.bedrock import _ConverseStream
from any_llm.types.completion import ChatCompletionChunk, CompletionParams
from any_llm.utils.aio import aclose_quietly


class _EventStream:
    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        block: bool = False,
        error: Exception | None = None,
        close_error: bool = False,
    ) -> None:
        self.events = iter(events)
        self.block = block
        self.error = error
        self.close_error = close_error
        self.read_started = threading.Event()
        self.release_read = threading.Event()
        self.closed = threading.Event()
        self.read_threads: list[int] = []
        self.close_threads: list[int] = []
        self.reading = False
        self.closed_during_read = False

    def __iter__(self) -> "_EventStream":
        return self

    def __next__(self) -> dict[str, Any]:
        self.read_threads.append(threading.get_ident())
        self.reading = True
        self.read_started.set()
        try:
            if self.block:
                assert self.release_read.wait(5), "The test did not release the SDK read."
            event = next(self.events, None)
            if event is not None:
                return event
            if self.error is not None:
                raise self.error
            raise StopIteration
        finally:
            self.reading = False

    def close(self) -> None:
        self.closed_during_read |= self.reading
        self.close_threads.append(threading.get_ident())
        self.closed.set()
        if self.close_error:
            message = "SDK close failed"
            raise RuntimeError(message)


def _text(text: str) -> dict[str, Any]:
    return {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}}


def _provider(event_stream: _EventStream) -> BedrockProvider:
    client = Mock()
    client.converse_stream.return_value = {"stream": event_stream}
    return BedrockProvider(client=client, unified_exceptions=True)


async def _open_stream(event_stream: _EventStream) -> AsyncIterator[ChatCompletionChunk]:
    return await _provider(event_stream).acompletion(
        model="amazon.nova-lite-v1:0", messages=[{"role": "user", "content": "Hello"}], stream=True
    )


async def _wait_until_set(event: threading.Event) -> None:
    # The queued-cleanup tests occupy the default executor, so offloading event.wait()
    # would prevent the test from reaching the code that releases its worker.
    async with asyncio.timeout(2):
        while not event.is_set():  # noqa: ASYNC110
            await asyncio.sleep(0.001)


async def _read_chunk(stream: AsyncIterator[ChatCompletionChunk]) -> ChatCompletionChunk:
    return await anext(stream)


@pytest.mark.asyncio
async def test_converse_stream_reads_off_loop_without_reading_ahead() -> None:
    raw = _EventStream([_text("first"), _text("second")], block=True)
    stream = await _open_stream(raw)
    loop_thread = threading.get_ident()
    assert not raw.read_threads
    read = asyncio.create_task(_read_chunk(stream))
    try:
        await _wait_until_set(raw.read_started)
        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), 1)
        assert not read.done()
        assert len(raw.read_threads) == 1
        assert raw.read_threads[0] != loop_thread
        raw.release_read.set()
        assert (await read).choices[0].delta.content == "first"
        await asyncio.sleep(0)
        assert len(raw.read_threads) == 1
        assert (await anext(stream)).choices[0].delta.content == "second"
        assert len(raw.read_threads) == 2
    finally:
        raw.release_read.set()
        await aclose_quietly(stream)
    assert len(raw.close_threads) == 1
    assert raw.close_threads[0] != loop_thread
    assert not raw.closed_during_read


@pytest.mark.asyncio
async def test_converse_stream_preserves_text_tool_order_and_indices() -> None:
    raw = _EventStream(
        [
            {"unknownEvent": {}},
            _text("hello"),
            {"contentBlockStart": {"contentBlockIndex": 3, "start": {"toolUse": {"toolUseId": "a", "name": "one"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 3, "delta": {"toolUse": {"input": "{}"}}}},
            {"contentBlockStart": {"contentBlockIndex": 7, "start": {"toolUse": {"toolUseId": "b", "name": "two"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 7, "delta": {"toolUse": {"input": '{"x":1}'}}}},
        ]
    )
    stream = await _open_stream(raw)
    chunks = [chunk async for chunk in stream]
    assert chunks[0].choices[0].delta.content == "hello"
    calls = [chunk.choices[0].delta.tool_calls for chunk in chunks[1:]]
    assert all(calls)
    assert [call[0].index for call in calls if call] == [0, 0, 1, 1]
    assert [call[0].function.arguments for call in calls if call and call[0].function] == ["", "{}", "", '{"x":1}']
    assert len(raw.close_threads) == 1


@pytest.mark.parametrize("events", [[], [{"unknownEvent": {}}], [_text("hello")]])
@pytest.mark.asyncio
async def test_converse_stream_exhaustion_closes_once(events: list[dict[str, Any]]) -> None:
    raw = _EventStream(events)
    stream = await _open_stream(raw)
    chunks = [chunk async for chunk in stream]
    assert len(chunks) == int(bool(events) and "contentBlockDelta" in events[0])
    await aclose_quietly(stream)
    assert len(raw.close_threads) == 1


@pytest.mark.parametrize("consume_first", [False, True])
@pytest.mark.asyncio
async def test_converse_stream_explicit_close_works_before_and_after_first_read(consume_first: bool) -> None:
    raw = _EventStream([_text("first"), _text("second")])
    stream = await _open_stream(raw)
    if consume_first:
        await anext(stream)
    await aclose_quietly(stream)
    await aclose_quietly(stream)
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert len(raw.read_threads) == int(consume_first)
    assert len(raw.close_threads) == 1


@pytest.mark.parametrize("close_error", [False, True])
@pytest.mark.asyncio
async def test_converse_stream_preserves_exception_conversion_and_closes(close_error: bool) -> None:
    error = EventStreamError(  # type: ignore[no-untyped-call]
        {"Error": {"Code": "ThrottlingException", "Message": "Too many requests"}}, "ConverseStream"
    )
    raw = _EventStream([_text("first")], error=error, close_error=close_error)
    stream = await _open_stream(raw)
    await anext(stream)
    with pytest.raises(RateLimitError) as raised:
        await anext(stream)
    assert raised.value.original_exception is error
    assert len(raw.close_threads) == 1
    await aclose_quietly(stream)
    assert len(raw.close_threads) == 1


@pytest.mark.asyncio
async def test_converse_stream_conversion_failure_closes() -> None:
    raw = _EventStream([{"messageStop": {}}])
    stream = await _open_stream(raw)
    with pytest.raises(AnyLLMError) as raised:
        await anext(stream)
    assert isinstance(raised.value.original_exception, KeyError)
    assert len(raw.close_threads) == 1


@pytest.mark.parametrize("cancel_read", [False, True])
@pytest.mark.parametrize("read_error", [False, True])
@pytest.mark.asyncio
async def test_converse_stream_defers_cleanup_to_the_active_read(cancel_read: bool, read_error: bool) -> None:
    raw = _EventStream([] if read_error else [_text("discarded")], block=True, error=RuntimeError("read failed"))
    stream = await _open_stream(raw)
    read = asyncio.create_task(_read_chunk(stream))
    try:
        await _wait_until_set(raw.read_started)
        if cancel_read:
            read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(read, 1)
        await asyncio.wait_for(aclose_quietly(stream), 1)
        await asyncio.wait_for(aclose_quietly(stream), 1)
        assert not raw.closed.is_set()
        assert raw.reading
    finally:
        raw.release_read.set()
    await _wait_until_set(raw.closed)
    if not cancel_read:
        with pytest.raises(AnyLLMError if read_error else StopAsyncIteration):
            await read
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert len(raw.read_threads) == 1
    assert len(raw.close_threads) == 1
    assert raw.close_threads[0] == raw.read_threads[0]
    assert not raw.closed_during_read


@pytest.mark.asyncio
async def test_converse_stream_rejects_concurrent_advances() -> None:
    raw = _EventStream([_text("first")], block=True)
    stream = _ConverseStream(raw, "model-id")
    read = asyncio.create_task(_read_chunk(stream))
    try:
        await _wait_until_set(raw.read_started)
        with pytest.raises(RuntimeError, match="already in progress"):
            await anext(stream)
        with pytest.raises(RuntimeError, match="already in progress"):
            next(stream)
        assert len(raw.read_threads) == 1
    finally:
        raw.release_read.set()
    assert (await read).choices[0].delta.content == "first"
    await stream.aclose()
    assert len(raw.close_threads) == 1


@pytest.mark.asyncio
async def test_converse_stream_cancelled_close_waiter_does_not_cancel_queued_cleanup() -> None:
    raw = _EventStream([_text("unused")])
    stream = _ConverseStream(raw, "model-id")
    occupied = threading.Event()
    release_worker = threading.Event()

    def occupy_worker() -> None:
        occupied.set()
        assert release_worker.wait(5), "The test did not release the executor."

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.get_running_loop().set_default_executor(executor)
        executor.submit(occupy_worker)
        try:
            await _wait_until_set(occupied)
            task = asyncio.create_task(stream.aclose())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not raw.closed.is_set()
        finally:
            release_worker.set()
        await _wait_until_set(raw.closed)
        await stream.aclose()
    assert not raw.read_threads
    assert len(raw.close_threads) == 1


@pytest.mark.asyncio
async def test_converse_stream_cancelled_queued_read_does_not_wait_for_cleanup() -> None:
    raw = _EventStream([_text("unused")])
    stream = await _open_stream(raw)
    occupied = threading.Event()
    release_worker = threading.Event()
    read: asyncio.Task[ChatCompletionChunk] | None = None

    def occupy_worker() -> None:
        occupied.set()
        assert release_worker.wait(5), "The test did not release the executor."

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.get_running_loop().set_default_executor(executor)
        executor.submit(occupy_worker)
        try:
            await _wait_until_set(occupied)
            read = asyncio.create_task(_read_chunk(stream))
            await asyncio.sleep(0)
            read.cancel()
            # wait() observes cancellation without issuing a second cancel that
            # could hide cleanup blocking in the public exception wrapper.
            done, _ = await asyncio.wait({read}, timeout=1)
            assert read in done, "Cancellation waited for the occupied executor."
            with pytest.raises(asyncio.CancelledError):
                await read
            assert not raw.closed.is_set()
        finally:
            release_worker.set()
            if read is not None and not read.done():
                with suppress(asyncio.CancelledError):
                    await read
        await _wait_until_set(raw.closed)
        await aclose_quietly(stream)
    assert not raw.read_threads
    assert len(raw.close_threads) == 1


@pytest.mark.asyncio
async def test_converse_stream_cancellation_while_opening_closes_the_late_response() -> None:
    raw = _EventStream([_text("unused")])
    opening = threading.Event()
    release_open = threading.Event()

    def open_stream(**kwargs: Any) -> dict[str, Any]:
        opening.set()
        assert release_open.wait(5), "The test did not release stream opening."
        return {"stream": raw}

    provider = _provider(raw)
    provider.client.converse_stream.side_effect = open_stream
    task = asyncio.create_task(
        provider.acompletion(
            model="amazon.nova-lite-v1:0", messages=[{"role": "user", "content": "Hello"}], stream=True
        )
    )
    try:
        await _wait_until_set(opening)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert not raw.closed.is_set()
    finally:
        release_open.set()
    await _wait_until_set(raw.closed)
    assert not raw.read_threads
    assert len(raw.close_threads) == 1
    assert raw.close_threads[0] != threading.get_ident()


@pytest.mark.parametrize("consume_first", [False, True])
def test_converse_sync_iterator_explicit_close(consume_first: bool) -> None:
    raw = _EventStream([_text("first"), _text("second")])
    stream = _provider(raw)._completion(
        CompletionParams(model_id="amazon.nova-lite-v1:0", messages=[{"role": "user", "content": "Hello"}], stream=True)
    )
    assert isinstance(stream, _ConverseStream)
    if consume_first:
        next(stream)
    stream.close()
    stream.close()
    assert list(stream) == []
    assert len(raw.read_threads) == int(consume_first)
    assert len(raw.close_threads) == 1


def test_converse_public_sync_stream_closes_early() -> None:
    raw = _EventStream([_text("first"), _text("second")])
    stream = _provider(raw).completion(
        model="amazon.nova-lite-v1:0", messages=[{"role": "user", "content": "Hello"}], stream=True
    )
    with closing(cast("Generator[ChatCompletionChunk, None, None]", stream)):
        assert next(stream).choices[0].delta.content == "first"
    # The sync bridge cancels its async consumer; its worker can finish cleanup after close returns.
    assert raw.closed.wait(1), "The sync bridge did not close its SDK stream."
    assert len(raw.close_threads) == 1
    assert not raw.closed_during_read


@pytest.mark.asyncio
async def test_converse_nonstreaming_completion_is_preserved() -> None:
    client = Mock()
    client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }
    result = await BedrockProvider(client=client).acompletion(
        model="amazon.nova-lite-v1:0", messages=[{"role": "user", "content": "Hello"}]
    )
    assert result.choices[0].message.content == "hello"
    client.converse_stream.assert_not_called()
