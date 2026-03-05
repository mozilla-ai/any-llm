"""Utilities for running async code in sync contexts."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import queue
import threading
from typing import TYPE_CHECKING, Any, TypeVar, cast

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator


def run_async_in_sync(coro: Coroutine[Any, Any, T], allow_running_loop: bool = True) -> T:
    """Run an async coroutine in a synchronous context.

    Handles different event loop scenarios:
    - If a loop is running, uses threading to avoid conflicts
    - If no loop exists, creates one or uses the current loop

    Args:
        coro: The coroutine to execute
        allow_running_loop: Whether to raise an error if called within a running event loop.

    Returns:
        The result of the coroutine execution

    """
    try:
        # Check if there's a running event loop
        asyncio.get_running_loop()
        running_loop = True
    except RuntimeError:
        running_loop = False

    if running_loop:
        if not allow_running_loop:
            msg = "Cannot use the `sync` API in an `async` context. Use the `async` API instead."
            raise RuntimeError(msg)

        # If we get here, there's a loop running, so we can't use run_until_complete()
        # or asyncio.run() - must use threading approach
        def run_in_thread() -> T:
            async def run_with_cleanup() -> T:
                try:
                    result = await coro

                    # Wait for any pending background tasks to complete to prevent "Event loop is closed" errors
                    pending_tasks = [
                        task for task in asyncio.all_tasks() if not task.done() and task is not asyncio.current_task()
                    ]

                    if pending_tasks:
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                except Exception:
                    pending_tasks = [
                        task for task in asyncio.all_tasks() if not task.done() and task is not asyncio.current_task()
                    ]

                    for task in pending_tasks:
                        task.cancel()

                    if pending_tasks:
                        await asyncio.gather(*pending_tasks, return_exceptions=True)

                    raise
                return result

            return asyncio.run(run_with_cleanup())

        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(run_in_thread).result()
    else:
        # No running event loop - try to get available loop
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop at all - create one
            return asyncio.run(coro)


def _async_source_to_sync_iter(
    get_async_iter: Callable[[], Awaitable[AsyncIterator[T]]], allow_running_loop: bool = True
) -> Iterator[T]:
    """Bridge an async iterator source into a synchronous iterator."""
    try:
        asyncio.get_running_loop()
        running_loop = True
    except RuntimeError:
        running_loop = False

    if running_loop and not allow_running_loop:
        msg = "Cannot use the `sync` API in an `async` context. Use the `async` API instead."
        raise RuntimeError(msg)

    done_sentinel = object()
    output_queue: queue.Queue[object] = queue.Queue()
    cancel_event = threading.Event()
    loop_ready = threading.Event()
    loop_holder: dict[str, asyncio.AbstractEventLoop | None] = {"loop": None}
    task_holder: dict[str, asyncio.Task[None] | None] = {"task": None}

    def worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder["loop"] = loop

        async def consume() -> None:
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

        task = loop.create_task(consume())
        task_holder["task"] = task
        loop_ready.set()

        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            output_queue.put(exc)
        finally:
            pending_tasks = [pending for pending in asyncio.all_tasks(loop) if not pending.done()]
            for pending in pending_tasks:
                pending.cancel()

            if pending_tasks:
                loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))

            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            output_queue.put(done_sentinel)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    loop_ready.wait()

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
        loop = loop_holder["loop"]
        task = task_holder["task"]
        if loop is not None and task is not None and not task.done():
            loop.call_soon_threadsafe(task.cancel)
        thread.join()


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
