import asyncio
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import MagicMock, Mock

import pytest
from botocore.exceptions import EventStreamError
from typing_extensions import override

from any_llm.exceptions import RateLimitError
from any_llm.providers.bedrock import BedrockProvider
from any_llm.types.completion import ChatCompletionChunk

MODEL = "amazon.nova-lite-v1:0"


class _FakeEventStream:
    """Stands in for botocore's ``EventStream``: iterable once, and closable.

    ``started`` is set from the worker thread at the beginning of a read, before ``release``
    is waited on, so a test can observe an in-flight read without sleeping.
    """

    def __init__(
        self,
        events: list[dict[str, Any]],
        error: Exception | None = None,
        *,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self._events = events
        self._error = error
        self._started = started
        self._release = release
        self._lock = threading.Lock()
        self._in_read = False
        self.read_threads: list[int] = []
        self.closed = False
        self.close_count = 0
        self.closed_while_reading = False
        self.closed_event = threading.Event()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for event in self._events:
            self._read_one()
            yield event
        if self._error is not None:
            self._read_one(error=self._error)

    def _read_one(self, error: Exception | None = None) -> None:
        with self._lock:
            self._in_read = True
        try:
            self.read_threads.append(threading.get_ident())
            if self._started is not None:
                self._started.set()
            if self._release is not None and not self._release.wait(5):
                msg = "fake Bedrock read was not released"
                raise TimeoutError(msg)
            if error is not None:
                raise error
        finally:
            with self._lock:
                self._in_read = False

    def close(self) -> None:
        with self._lock:
            if self._in_read:
                self.closed_while_reading = True
            self.close_count += 1
            self.closed = True
        self.closed_event.set()


def _text(text: str) -> dict[str, Any]:
    return {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}}


async def _open_stream(
    event_stream: _FakeEventStream, response_metadata: dict[str, Any] | None = None
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """Return the public stream. The cast is for ``aclose``, which the runtime wrapper has."""
    client = Mock()
    client.converse_stream.return_value = {"stream": event_stream, "ResponseMetadata": response_metadata or {}}
    provider = BedrockProvider(client=client, unified_exceptions=True)
    stream = await provider.acompletion(model=MODEL, messages=[{"role": "user", "content": "Hello"}], stream=True)
    assert isinstance(stream, AsyncIterator)
    return cast("AsyncGenerator[ChatCompletionChunk, None]", stream)


async def _next_chunk(stream: AsyncIterator[ChatCompletionChunk]) -> ChatCompletionChunk:
    return await anext(stream)


class _CountingExecutor(ThreadPoolExecutor):
    """One-worker pool that tracks submitted jobs and their futures."""

    def __init__(self) -> None:
        super().__init__(max_workers=1, thread_name_prefix="bedrock-queued")
        self.submitted = 0
        self.futures: list[Future[Any]] = []
        self.second_job = asyncio.Event()

    @override
    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        self.submitted += 1
        future: Future[Any] = super().submit(fn, *args, **kwargs)
        self.futures.append(future)
        if self.submitted == 2:
            self.second_job.set()
        return future


@asynccontextmanager
async def _busy_single_worker_executor() -> AsyncIterator[tuple[_CountingExecutor, threading.Event]]:
    """Install a one-worker default executor and hold its only worker until ``release`` is set.

    ``asyncio.to_thread`` uses the loop's default executor. A job submitted while that worker is
    inside ``occupy`` stays queued, which is the cancellation window these tests reproduce.
    """
    loop = asyncio.get_running_loop()
    raw_executor: object = getattr(loop, "_default_executor", None)
    previous = raw_executor if isinstance(raw_executor, ThreadPoolExecutor) else None
    executor = _CountingExecutor()
    loop.set_default_executor(executor)
    entered = asyncio.Event()
    release = threading.Event()

    def occupy() -> None:
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            msg = "test left the executor worker blocked"
            raise TimeoutError(msg)

    occupier = asyncio.create_task(asyncio.to_thread(occupy))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        yield executor, release
    finally:
        release.set()
        await asyncio.gather(occupier, return_exceptions=True)
        if previous is not None:
            loop.set_default_executor(previous)
        else:
            loop.set_default_executor(ThreadPoolExecutor(thread_name_prefix="asyncio"))
        executor.shutdown(wait=True, cancel_futures=True)


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
    assert event_stream.close_count == 1
    assert not event_stream.closed_while_reading


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
    # A client joins a call's deltas by id, so only the first delta names the call, as OpenAI streams it.
    assert [calls[0].id for calls in tool_calls if calls] == ["a", None, "b", None]
    assert [calls[0].function.name for calls in tool_calls if calls and calls[0].function] == ["one", None, "two", None]
    assert event_stream.close_count == 1
    assert not event_stream.closed_while_reading


@pytest.mark.asyncio
async def test_converse_stream_chunks_share_the_aws_request_id_and_a_creation_time() -> None:
    """AWS puts no id in the chunks, so each one used to get its own, which broke grouping a stream by id."""
    event_stream = _FakeEventStream(
        [
            {"messageStart": {"role": "assistant"}},
            _text("hi"),
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {}}},
        ]
    )
    stream = await _open_stream(event_stream, {"RequestId": "req-1"})
    chunks = [chunk async for chunk in stream]

    assert {chunk.id for chunk in chunks} == {"req-1"}
    assert len({chunk.created for chunk in chunks}) == 1
    assert chunks[0].created > 0


@pytest.mark.asyncio
async def test_converse_stream_reports_usage_on_a_chunk_with_no_choices() -> None:
    """The usage arrives after the finish, on a chunk of its own with no choices, as OpenAI sends it."""
    event_stream = _FakeEventStream(
        [
            _text("hi"),
            {"messageStop": {"stopReason": "max_tokens"}},
            {"metadata": {"usage": {"inputTokens": 18, "outputTokens": 5, "totalTokens": 23}, "metrics": {}}},
        ]
    )
    stream = await _open_stream(event_stream)
    chunks = [chunk async for chunk in stream]

    finish, usage = chunks[-2:]
    assert finish.choices[0].finish_reason == "length"
    assert finish.usage is None
    assert usage.choices == []
    assert usage.usage is not None
    assert (usage.usage.prompt_tokens, usage.usage.completion_tokens, usage.usage.total_tokens) == (18, 5, 23)


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
    assert event_stream.close_count == 1
    assert not event_stream.closed_while_reading


@pytest.mark.asyncio
async def test_converse_stream_closed_early_closes_the_bedrock_stream() -> None:
    event_stream = _FakeEventStream([_text("first"), _text("second")])

    stream = BedrockProvider._stream_converse(event_stream, MODEL, "req-1")
    await anext(stream)
    await stream.aclose()

    assert len(event_stream.read_threads) == 1
    assert event_stream.close_count == 1
    assert not event_stream.closed_while_reading


@pytest.mark.asyncio
async def test_converse_stream_closed_before_first_read_closes_once() -> None:
    event_stream = _FakeEventStream([_text("first"), _text("second")])
    stream = BedrockProvider._stream_converse(event_stream, MODEL, "req-1")

    await stream.aclose()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)

    assert event_stream.read_threads == []
    assert event_stream.close_count == 1
    await stream.aclose()
    assert event_stream.close_count == 1
    assert not event_stream.closed_while_reading


@pytest.mark.asyncio
async def test_public_converse_stream_closed_before_first_read_closes_once() -> None:
    event_stream = _FakeEventStream([_text("first")])
    stream = await _open_stream(event_stream)

    await stream.aclose()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)

    assert event_stream.read_threads == []
    assert event_stream.close_count == 1
    await stream.aclose()
    assert event_stream.close_count == 1
    assert not event_stream.closed_while_reading


@pytest.mark.asyncio
async def test_converse_stream_heartbeat_runs_during_a_slow_read() -> None:
    """Other tasks keep running while a Converse read blocks on a worker thread."""
    started = threading.Event()
    release = threading.Event()
    event_stream = _FakeEventStream([_text("late")], started=started, release=release)
    stream = await _open_stream(event_stream)
    beats = 0

    async def heartbeat() -> None:
        nonlocal beats
        while True:
            beats += 1
            await asyncio.sleep(0)

    heart: asyncio.Task[None] = asyncio.create_task(heartbeat())
    read: asyncio.Task[ChatCompletionChunk] = asyncio.create_task(_next_chunk(stream))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        for _ in range(5):
            await asyncio.sleep(0)
        assert beats >= 5
        assert not read.done()
        assert event_stream.read_threads
        assert event_stream.read_threads[0] != threading.get_ident()
        release.set()
        assert (await read).choices[0].delta.content == "late"
    finally:
        release.set()
        heart.cancel()
        read.cancel()
        await asyncio.gather(heart, read, return_exceptions=True)
        await stream.aclose()

    assert event_stream.close_count == 1
    assert not event_stream.closed_while_reading


@pytest.mark.asyncio
async def test_converse_stream_cancel_during_read_closes_once_after_the_read() -> None:
    """A cancelled read stays inside ``next`` until it returns, then the raw stream is closed once."""
    started = threading.Event()
    release = threading.Event()
    event_stream = _FakeEventStream([_text("late"), _text("more")], started=started, release=release)
    stream = await _open_stream(event_stream)
    read: asyncio.Task[ChatCompletionChunk] = asyncio.create_task(_next_chunk(stream))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        read.cancel()
        for _ in range(20):
            await asyncio.sleep(0)
        assert not read.done()
        assert event_stream.close_count == 0
        assert not event_stream.closed_while_reading
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await read
        assert event_stream.close_count == 1
        assert not event_stream.closed_while_reading
    finally:
        release.set()
        if not read.done():
            read.cancel()
            await asyncio.gather(read, return_exceptions=True)
        await stream.aclose()


@pytest.mark.asyncio
async def test_converse_stream_cancel_queued_read_finishes_and_closes_with_no_reads() -> None:
    """Cancelling a read that is only queued must finish, and must close the raw stream.

    The default executor has one worker and that worker is busy, so ``__anext__`` queues
    ``blocking_next``. Cancelling the task drops that queued job. Cleanup still has to finish,
    with the read never having started.
    """
    event_stream = _FakeEventStream([_text("first"), _text("second")])
    stream = BedrockProvider._stream_converse(event_stream, MODEL, "req-1")
    async with _busy_single_worker_executor() as (executor, release):

        async def wait_for_read_cancellation() -> None:
            while not executor.futures[1].cancelled():  # noqa: ASYNC110
                await asyncio.sleep(0)

        read = asyncio.create_task(_next_chunk(stream))
        try:
            await asyncio.wait_for(executor.second_job.wait(), timeout=2)
            assert executor.submitted == 2
            assert not read.done()
            assert event_stream.read_threads == []
            assert not event_stream.closed

            read.cancel()
            await asyncio.wait_for(wait_for_read_cancellation(), timeout=2)
            release.set()
            done, _pending = await asyncio.wait({read}, timeout=2)
            assert read in done
            with pytest.raises(asyncio.CancelledError):
                read.result()
            assert event_stream.close_count == 1
            assert event_stream.read_threads == []
            assert not event_stream.closed_while_reading
        finally:
            release.set()
            if not read.done():
                # A regressed cleanup waits forever for a read that will never run. Wake that
                # wait so the worker can exit and this failure stays an assertion, not a hang.
                stream._events._closed.set()
                await asyncio.wait({read}, timeout=2)
            await stream.aclose()


@pytest.mark.asyncio
async def test_converse_stream_cancelled_aclose_still_closes_and_a_later_aclose_waits() -> None:
    """Cancelling ``aclose`` while ``close_raw`` is only queued must not abandon the raw stream.

    The close used to be recorded before the queued call ran. Dropping that call left the stream
    open, and the next ``aclose`` returned without closing it. The later call has to wait for the
    same close, which cannot finish until the busy worker is released.
    """
    event_stream = _FakeEventStream([_text("first")])
    stream = BedrockProvider._stream_converse(event_stream, MODEL, "req-1")
    async with _busy_single_worker_executor() as (executor, release):
        closing = asyncio.create_task(stream.aclose())
        try:
            await asyncio.wait_for(executor.second_job.wait(), timeout=2)
            assert executor.submitted == 2
            assert not closing.done()
            assert not event_stream.closed
            assert event_stream.read_threads == []

            closing.cancel()
            done, _pending = await asyncio.wait({closing}, timeout=2)
            assert closing in done
            with pytest.raises(asyncio.CancelledError):
                closing.result()
            # The only worker is still inside the test's blocker, so a dropped close cannot have run.
            assert not event_stream.closed

            again = asyncio.create_task(stream.aclose())
            for _ in range(5):
                await asyncio.sleep(0)
            assert not again.done()
            assert not event_stream.closed

            release.set()
            done, _pending = await asyncio.wait({again}, timeout=2)
            assert again in done
            assert again.exception() is None
            assert event_stream.close_count == 1
            assert executor.submitted == 2
            assert event_stream.read_threads == []
            assert not event_stream.closed_while_reading
            await stream.aclose()
            assert event_stream.close_count == 1
        finally:
            release.set()
            if not closing.done():
                closing.cancel()
                await asyncio.wait({closing}, timeout=2)
            await stream.aclose()


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
    stream = BedrockProvider._stream_converse(event_stream, MODEL, "req-1")
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
    assert event_stream.close_count == 1
    assert not event_stream.closed_while_reading
