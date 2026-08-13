import asyncio
import contextvars
import threading
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


def test_run_async_in_sync_cleans_up_background_tasks_without_running_loop() -> None:
    task_completed = {"value": False}
    result: list[str] = []
    errors: list[Exception] = []

    async def operation_with_background_task() -> str:
        async def background_work() -> None:
            await asyncio.sleep(0)
            task_completed["value"] = True

        task = asyncio.create_task(background_work())
        assert task is not None
        return "operation_started"

    def run_in_thread() -> None:
        try:
            result.append(run_async_in_sync(operation_with_background_task()))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert result == ["operation_started"]
    assert task_completed["value"] is True


def test_run_async_in_sync_cancels_its_background_tasks_when_the_operation_fails() -> None:
    background_cancelled = threading.Event()
    errors: list[Exception] = []

    async def failing_operation() -> str:
        started = asyncio.Event()

        async def background_work() -> None:
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                background_cancelled.set()
                raise

        task = asyncio.create_task(background_work())
        assert task is not None
        await started.wait()

        msg = "provider failed"
        raise ValueError(msg)

    def run_in_thread() -> None:
        try:
            run_async_in_sync(failing_operation())
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert [str(error) for error in errors] == ["provider failed"]
    # This loop is about to close, so its leftover task is cancelled while it can still clean up.
    assert background_cancelled.is_set()


def test_run_async_in_sync_disallows_running_loop_when_requested() -> None:
    async def operation() -> str:
        return "unreachable"

    async def call_from_async_context() -> None:
        coro = operation()
        with pytest.raises(RuntimeError, match="Cannot use the `sync` API in an `async` context"):
            run_async_in_sync(coro, allow_running_loop=False)
        coro.close()

    asyncio.run(call_from_async_context())


def test_run_async_in_sync_leaves_caller_owned_tasks_on_a_reused_loop() -> None:
    state: dict[str, Any] = {}

    def run_in_thread() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def caller_background_work() -> None:
            await asyncio.sleep(30)

        caller_task = loop.create_task(caller_background_work())

        async def operation() -> str:
            await asyncio.sleep(0)
            return "operation_finished"

        async def failing_operation() -> str:
            await asyncio.sleep(0)
            msg = "provider failed"
            raise ValueError(msg)

        # A loop handed back by `asyncio.get_event_loop()` is the caller's, and its tasks may never
        # finish, so waiting on them would hang and cancelling them would discard the caller's work.
        state["result"] = run_async_in_sync(operation())
        with pytest.raises(ValueError, match="provider failed"):
            run_async_in_sync(failing_operation())
        state["caller_task_cancelled"] = caller_task.cancelled()

        caller_task.cancel()
        loop.run_until_complete(asyncio.gather(caller_task, return_exceptions=True))
        loop.close()

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive(), "sync call blocked on a task it does not own"
    assert state["result"] == "operation_finished"
    assert state["caller_task_cancelled"] is False


def test_run_async_in_sync_ignores_a_closed_current_loop() -> None:
    results: list[str] = []

    def run_in_thread() -> None:
        closed_loop = asyncio.new_event_loop()
        closed_loop.close()
        asyncio.set_event_loop(closed_loop)

        async def operation() -> str:
            return "ran on a fresh loop"

        results.append(run_async_in_sync(operation()))

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert results == ["ran on a fresh loop"]


def test_run_async_in_sync_propagates_runtime_error_from_coroutine() -> None:
    errors: list[Exception] = []

    def run_in_thread() -> None:
        asyncio.set_event_loop(asyncio.new_event_loop())

        async def failing_operation() -> str:
            await asyncio.sleep(0)
            msg = "provider raised RuntimeError"
            raise RuntimeError(msg)

        # Looking the loop up separately from running the coroutine keeps a RuntimeError raised by
        # the coroutine from being mistaken for a missing loop and retried on a spent coroutine.
        try:
            run_async_in_sync(failing_operation())
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert [str(error) for error in errors] == ["provider raised RuntimeError"]


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
