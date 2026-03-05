import asyncio
import contextvars
import time
from collections.abc import AsyncIterator, Generator
from typing import Any, cast

import pytest

from any_llm.utils.aio import async_iter_to_sync_iter, run_async_in_sync


def test_run_async_in_sync_fails_with_background_task_state() -> None:
    task_completed = {"value": False}

    async def operation_with_critical_background_task() -> str:
        """Simulates an operation where a background task MUST complete for success."""

        async def critical_background_work() -> None:
            await asyncio.sleep(0.02)
            task_completed["value"] = True

        task = asyncio.create_task(critical_background_work())
        assert task is not None
        return "operation_started"

    async def test_in_streamlit_context() -> None:
        task_completed["value"] = False
        # This triggers the threading in  run_async_in_sync
        result = run_async_in_sync(operation_with_critical_background_task())
        assert result == "operation_started"
        await asyncio.sleep(0.05)
        assert task_completed["value"] is True

    asyncio.run(test_in_streamlit_context())


def test_async_iter_to_sync_iter_preserves_contextvars() -> None:
    current_context = contextvars.ContextVar("current_context", default="unset")

    async def source() -> AsyncIterator[str]:
        token = current_context.set("active")
        try:
            yield "one"
            yield "two"
        finally:
            current_context.reset(token)

    chunks = list(async_iter_to_sync_iter(source()))

    assert chunks == ["one", "two"]
    assert current_context.get() == "unset"


def test_async_iter_to_sync_iter_closes_cleanly_on_generator_close() -> None:
    cleanup = {"done": False}

    async def source() -> AsyncIterator[int]:
        try:
            yield 1
            await asyncio.sleep(10)
        finally:
            cleanup["done"] = True

    iterator = cast("Generator[int, Any, None]", async_iter_to_sync_iter(source()))

    assert next(iterator) == 1
    iterator.close()

    deadline = time.time() + 2
    while time.time() < deadline and not cleanup["done"]:
        time.sleep(0.01)

    assert cleanup["done"] is True


def test_async_iter_to_sync_iter_disallows_running_loop_when_requested() -> None:
    async def source() -> AsyncIterator[int]:
        yield 1

    async def consume_in_async_context() -> None:
        with pytest.raises(RuntimeError, match="Cannot use the `sync` API in an `async` context"):
            list(async_iter_to_sync_iter(source(), allow_running_loop=False))

    asyncio.run(consume_in_async_context())
