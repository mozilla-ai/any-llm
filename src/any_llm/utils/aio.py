"""Utilities for running async code in sync contexts."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Any, TypeVar

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine, Iterator


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
            return asyncio.run(coro)

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


def async_iter_to_sync_iter(async_iter: AsyncIterator[T]) -> Iterator[T]:
    """Convert async iterable to sync iterable using a generator approach."""
    while True:
        try:
            awaitable = async_iter.__anext__()
            if not asyncio.iscoroutine(awaitable):
                msg = "awaitable is not a coroutine"
                raise ValueError(msg)
            yield run_async_in_sync(awaitable)
        except StopAsyncIteration:
            break
