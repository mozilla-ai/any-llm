"""Utilities for running async code in sync contexts."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import os
import queue
import threading
from typing import TYPE_CHECKING, Any, TypeVar, cast

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator

RUNNER_THREAD_NAME = "any-llm-async-runner"


def _start_loop_thread(name: str) -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Start an event loop in a new daemon thread and return both once the loop is running."""
    loop = asyncio.new_event_loop()
    loop_started = threading.Event()

    def run_forever() -> None:
        asyncio.set_event_loop(loop)
        loop_started.set()
        loop.run_forever()

    thread = threading.Thread(target=run_forever, name=name, daemon=True)
    thread.start()
    loop_started.wait()

    return loop, thread


class _RunnerLoop:
    """Holds the long-lived event loop that backs the sync API.

    Provider clients are built once per provider instance and cache their transports (HTTP
    connection pools, gRPC channels), which stay bound to the loop that first used them. Running
    each sync call on its own loop and then closing it leaves those transports bound to a dead
    loop, so the next call fails with "Event loop is closed". A loop that outlives every call
    keeps them usable.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                self._loop, _ = _start_loop_thread(RUNNER_THREAD_NAME)

            return self._loop

    def forget(self) -> None:
        """Drop the loop inherited by a forked child, whose thread did not survive the fork."""
        self._lock = threading.Lock()
        self._loop = None


_runner = _RunnerLoop()

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_runner.forget)


def _get_runner_loop() -> asyncio.AbstractEventLoop:
    """Return the long-lived event loop that backs the sync API."""
    return _runner.get()


def _pending_tasks() -> list[asyncio.Task[Any]]:
    """Return the unfinished tasks on the running loop, other than the one asking."""
    return [task for task in asyncio.all_tasks() if not task.done() and task is not asyncio.current_task()]


async def _await_with_cleanup(coro: Coroutine[Any, Any, T]) -> T:
    """Await ``coro``, then settle every remaining task before the loop it runs on is closed.

    Only safe on a loop dedicated to this one call: every task on it belongs to this call. Async
    HTTP clients can schedule response cleanup after the awaited coroutine returns, and closing the
    loop first breaks it.
    """
    try:
        result = await coro
    except Exception:
        pending_tasks = _pending_tasks()

        for task in pending_tasks:
            task.cancel()

        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        raise

    pending_tasks = _pending_tasks()
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    return result


async def _settle_remaining_tasks() -> None:
    """Cancel and await whatever is left on the running loop, so it can be closed safely.

    Cancelling on a live loop lets each task run its own cleanup, which is what keeps a transport
    from trying to schedule callbacks on a loop that is already gone.
    """
    pending_tasks = _pending_tasks()

    for task in pending_tasks:
        task.cancel()

    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    await asyncio.get_running_loop().shutdown_asyncgens()


def _submit_to_loop(coro: Coroutine[Any, Any, T], loop: asyncio.AbstractEventLoop) -> T:
    """Run ``coro`` on ``loop`` from a thread that is not running it, and block for the result.

    Background tasks the coroutine leaves behind are not waited on or cancelled: the loop is shared
    with other calls and outlives all of them, so those tasks finish on their own and are not this
    call's to touch.
    """
    context = contextvars.copy_context()
    result_future: concurrent.futures.Future[T] = concurrent.futures.Future()
    task_holder: dict[str, asyncio.Task[T]] = {}

    def schedule() -> None:
        task = loop.create_task(coro, context=context)
        task_holder["task"] = task
        task.add_done_callback(complete)

    def complete(task: asyncio.Task[T]) -> None:
        if task.cancelled():
            result_future.set_exception(asyncio.CancelledError())
            return

        error = task.exception()
        if error is not None:
            result_future.set_exception(error)
        else:
            result_future.set_result(task.result())

    loop.call_soon_threadsafe(schedule)

    try:
        return result_future.result()
    except BaseException:
        # An interrupt in the calling thread should not leave the coroutine running on the loop.
        task = task_holder.get("task")
        if task is not None and not task.done():
            loop.call_soon_threadsafe(task.cancel)
        raise


def _stop_loop_thread(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    """Settle the work left on a private loop, then stop its thread and close it."""
    with contextlib.suppress(Exception):
        # Teardown runs from a `finally`, so a failure to settle must not mask the original error.
        _submit_to_loop(_settle_remaining_tasks(), loop)

    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.close()


def _run_in_dedicated_loop(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` on a throwaway loop in a new thread.

    Only used when the caller is itself running on the runner loop, where submitting back to that
    loop and blocking on the result would deadlock.
    """

    def run_in_thread() -> T:
        return asyncio.run(_await_with_cleanup(coro))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        return executor.submit(run_in_thread).result()


def _is_running_on(loop: asyncio.AbstractEventLoop) -> bool:
    """Whether the calling thread is the one driving ``loop``."""
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def _reject_running_loop_if_needed(allow_running_loop: bool) -> asyncio.AbstractEventLoop | None:
    """Return the loop running in this thread, if any, rejecting the call when not allowed."""
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    if not allow_running_loop:
        msg = "Cannot use the `sync` API in an `async` context. Use the `async` API instead."
        raise RuntimeError(msg)

    return running_loop


def run_async_in_sync(coro: Coroutine[Any, Any, T], allow_running_loop: bool = True) -> T:
    """Run an async coroutine in a synchronous context.

    The coroutine runs on a long-lived event loop owned by any-llm, so cached provider clients
    keep working across calls, and the calling thread blocks until it finishes.

    Args:
        coro: The coroutine to execute
        allow_running_loop: Whether to raise an error if called within a running event loop.

    Returns:
        The result of the coroutine execution

    """
    running_loop = _reject_running_loop_if_needed(allow_running_loop)
    runner_loop = _get_runner_loop()

    if running_loop is runner_loop:
        return _run_in_dedicated_loop(coro)

    return _submit_to_loop(coro, runner_loop)


def _async_source_to_sync_iter(
    get_async_iter: Callable[[], Awaitable[AsyncIterator[T]]], allow_running_loop: bool = True
) -> Iterator[T]:
    """Bridge an async iterator source into a synchronous iterator."""
    running_loop = _reject_running_loop_if_needed(allow_running_loop)

    # The whole iterator is consumed by one task so the source keeps a single context, which async
    # generators rely on to reset the contextvar tokens they set.
    runner_loop = _get_runner_loop()
    if running_loop is runner_loop:
        # Draining the queue below would block the runner loop's own thread, so the consumer could
        # never make progress. This nested case gets a private loop instead.
        loop, loop_thread = _start_loop_thread(f"{RUNNER_THREAD_NAME}-nested")
    else:
        loop, loop_thread = runner_loop, None

    done_sentinel = object()
    output_queue: queue.Queue[object] = queue.Queue()
    cancel_event = threading.Event()
    task_ready = threading.Event()
    consumer_done = threading.Event()
    task_holder: dict[str, asyncio.Task[None] | None] = {"task": None}

    async def consume() -> None:
        try:
            async_iter = await get_async_iter()
            try:
                async for item in async_iter:
                    if cancel_event.is_set():
                        break
                    output_queue.put(item)
            finally:
                aclose = getattr(async_iter, "aclose", None)
                if callable(aclose):
                    with contextlib.suppress(Exception):
                        maybe_awaitable = aclose()
                        if asyncio.iscoroutine(maybe_awaitable):
                            await maybe_awaitable
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            output_queue.put(exc)
        finally:
            output_queue.put(done_sentinel)

    context = contextvars.copy_context()

    def schedule() -> None:
        task = loop.create_task(consume(), context=context)
        task.add_done_callback(lambda _: consumer_done.set())
        task_holder["task"] = task
        task_ready.set()

    loop.call_soon_threadsafe(schedule)
    # Both waits below are unbounded, as the thread-based implementation before them was. Nothing
    # stops the loop being submitted to, so `schedule` always runs, and `consume` always settles the
    # source before signalling, so the only way to wait forever is a source that never finishes
    # closing, which no timeout here could rescue.
    task_ready.wait()

    try:
        while True:
            result = output_queue.get()
            if result is done_sentinel:
                break
            if isinstance(result, Exception):
                raise result
            yield cast("T", result)
    finally:
        cancel_event.set()
        task = task_holder["task"]
        if task is not None and not task.done():
            loop.call_soon_threadsafe(task.cancel)

        # Closing from the thread that drives the loop, which happens when a garbage collection
        # pass finalises an abandoned iterator there, must not wait: the cancellation just
        # scheduled can only run once this thread returns to the loop, so blocking would wedge it
        # permanently. Leave the consumer to settle on its own in that case.
        if not _is_running_on(loop):
            # `consume` closes the source, so waiting for it means the source is fully cleaned up
            # by the time this iterator returns.
            consumer_done.wait()

            if loop_thread is not None:
                _stop_loop_thread(loop, loop_thread)


def async_iter_to_sync_iter(async_iter: AsyncIterator[T], allow_running_loop: bool = True) -> Iterator[T]:
    """Convert an async iterator into a sync iterator."""

    async def get_async_iter() -> AsyncIterator[T]:
        return async_iter

    return _async_source_to_sync_iter(get_async_iter, allow_running_loop=allow_running_loop)


def async_coro_to_sync_iter(
    async_iter_coro: Coroutine[Any, Any, AsyncIterator[T]], allow_running_loop: bool = True
) -> Iterator[T]:
    """Convert a coroutine returning an async iterator into a sync iterator."""

    async def get_async_iter() -> AsyncIterator[T]:
        return await async_iter_coro

    return _async_source_to_sync_iter(get_async_iter, allow_running_loop=allow_running_loop)
