import asyncio
import contextvars
import threading
import time
from collections.abc import AsyncIterator, Generator
from typing import Any, cast

import pytest

from any_llm.utils.aio import _get_runner_loop, _runner, async_iter_to_sync_iter, run_async_in_sync


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


def test_run_async_in_sync_background_tasks_survive_the_call_without_a_running_loop() -> None:
    task_completed = threading.Event()
    result: list[str] = []
    errors: list[Exception] = []

    async def operation_with_background_task() -> str:
        async def background_work() -> None:
            await asyncio.sleep(0.02)
            task_completed.set()

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
    # The runner loop outlives the call, so the task keeps running instead of being cancelled.
    assert task_completed.wait(timeout=5)


def test_run_async_in_sync_reuses_one_open_loop_across_calls() -> None:
    loops: list[asyncio.AbstractEventLoop] = []

    async def record_loop() -> str:
        loops.append(asyncio.get_running_loop())
        return "recorded"

    def run_in_thread() -> None:
        run_async_in_sync(record_loop())
        run_async_in_sync(record_loop())

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(loops) == 2
    # Provider clients cache transports bound to the loop that created them, so every sync call has
    # to run on the same loop, and that loop must stay open once the call returns.
    assert loops[0] is loops[1]
    assert not loops[0].is_closed()


def test_run_async_in_sync_does_not_wait_for_background_tasks_of_other_calls() -> None:
    long_running_started = threading.Event()

    async def operation_leaving_a_long_task() -> str:
        async def long_running_work() -> None:
            long_running_started.set()
            await asyncio.sleep(30)

        task = asyncio.create_task(long_running_work())
        assert task is not None
        return "first"

    async def second_operation() -> str:
        return "second"

    def run_in_thread() -> None:
        run_async_in_sync(operation_leaving_a_long_task())
        assert long_running_started.wait(timeout=5)
        run_async_in_sync(second_operation())

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=10)

    assert not thread.is_alive(), "a sync call blocked on background work it does not own"


@pytest.fixture
def restore_runner_loop() -> Generator[None, None, None]:
    """Undo runner loop swaps, so tests holding clients bound to it are not left stranded."""
    previous_loop = _runner._loop
    try:
        yield
    finally:
        _runner._loop = previous_loop


def test_runner_loop_is_rebuilt_after_a_fork(restore_runner_loop: None) -> None:
    first_loop = _get_runner_loop()

    # A forked child inherits the loop object but not the thread running it, so the child has to
    # build its own instead of waiting on a loop nothing is driving.
    _runner.forget()
    second_loop = _get_runner_loop()

    assert second_loop is not first_loop

    async def operation() -> str:
        return "ran on the new loop"

    result: list[str] = []
    thread = threading.Thread(target=lambda: result.append(run_async_in_sync(operation())), daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result == ["ran on the new loop"]


def test_runner_loop_is_replaced_once_it_has_been_closed(restore_runner_loop: None) -> None:
    stale_loop = asyncio.new_event_loop()
    stale_loop.close()
    _runner._loop = stale_loop

    assert _get_runner_loop() is not stale_loop


def test_run_async_in_sync_ignores_a_closed_current_loop() -> None:
    results: list[str] = []

    def run_in_thread() -> None:
        closed_loop = asyncio.new_event_loop()
        closed_loop.close()
        asyncio.set_event_loop(closed_loop)

        async def operation() -> str:
            return "ran on the runner loop"

        results.append(run_async_in_sync(operation()))

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert results == ["ran on the runner loop"]


def test_run_async_in_sync_from_the_runner_loop_does_not_deadlock() -> None:
    async def inner() -> str:
        return "inner"

    async def outer() -> str:
        # A coroutine already running on the runner loop calling back into the sync API.
        return run_async_in_sync(inner())

    def run_in_thread() -> None:
        assert run_async_in_sync(outer()) == "inner"

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=10)

    assert not thread.is_alive()


def test_nested_sync_call_settles_its_background_tasks_before_its_loop_closes() -> None:
    background_finished = threading.Event()
    background_cancelled = threading.Event()

    async def succeeding_inner() -> str:
        async def background_work() -> None:
            await asyncio.sleep(0.01)
            background_finished.set()

        task = asyncio.create_task(background_work())
        assert task is not None
        return "inner"

    async def failing_inner() -> str:
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

        msg = "inner failed"
        raise ValueError(msg)

    async def outer() -> str:
        # These nested calls get a private loop that really does close, so leftover tasks have to be
        # settled first: awaited on success, cancelled on failure.
        result = run_async_in_sync(succeeding_inner())
        with pytest.raises(ValueError, match="inner failed"):
            run_async_in_sync(failing_inner())
        return result

    results: list[str] = []
    thread = threading.Thread(target=lambda: results.append(run_async_in_sync(outer())), daemon=True)
    thread.start()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert results == ["inner"]
    assert background_finished.is_set()
    assert background_cancelled.is_set()


def test_run_async_in_sync_surfaces_cancellation_of_its_own_task() -> None:
    async def cancel_itself() -> str:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await asyncio.sleep(0)
        return "unreachable"

    errors: list[BaseException] = []

    def run_in_thread() -> None:
        try:
            run_async_in_sync(cancel_itself())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert [type(error) for error in errors] == [asyncio.CancelledError]


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


def test_run_async_in_sync_propagates_runtime_error_from_coroutine() -> None:
    errors: list[Exception] = []

    def run_in_thread() -> None:
        asyncio.set_event_loop(asyncio.new_event_loop())

        async def failing_operation() -> str:
            await asyncio.sleep(0)
            msg = "provider raised RuntimeError"
            raise RuntimeError(msg)

        try:
            run_async_in_sync(failing_operation())
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert [str(error) for error in errors] == ["provider raised RuntimeError"]


def test_run_async_in_sync_disallows_running_loop_when_requested() -> None:
    async def operation() -> str:
        return "unreachable"

    async def call_from_async_context() -> None:
        coro = operation()
        with pytest.raises(RuntimeError, match="Cannot use the `sync` API in an `async` context"):
            run_async_in_sync(coro, allow_running_loop=False)
        coro.close()

    asyncio.run(call_from_async_context())


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


def test_async_iter_to_sync_iter_shares_the_open_runner_loop() -> None:
    loops: list[asyncio.AbstractEventLoop] = []

    async def record_loop() -> str:
        loops.append(asyncio.get_running_loop())
        return "recorded"

    async def source() -> AsyncIterator[str]:
        loops.append(asyncio.get_running_loop())
        yield "chunk"

    def run_in_thread() -> None:
        run_async_in_sync(record_loop())
        assert list(async_iter_to_sync_iter(source())) == ["chunk"]

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(loops) == 2
    # Streaming and non-streaming sync calls share the provider client, so they have to share the
    # loop its transports are bound to, and that loop must survive the iterator finishing.
    assert loops[0] is loops[1]
    assert not loops[0].is_closed()


def test_async_iter_to_sync_iter_propagates_source_errors() -> None:
    async def source() -> AsyncIterator[int]:
        yield 1
        msg = "source failed"
        raise ValueError(msg)

    iterator = async_iter_to_sync_iter(source())

    assert next(iterator) == 1
    with pytest.raises(ValueError, match="source failed"):
        next(iterator)


def test_async_iter_to_sync_iter_from_the_runner_loop_uses_a_private_loop() -> None:
    source_loops: list[asyncio.AbstractEventLoop] = []
    background_cancelled = threading.Event()

    async def source() -> AsyncIterator[int]:
        source_loops.append(asyncio.get_running_loop())
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
        yield 1
        yield 2

    async def outer() -> list[int]:
        # Consuming an iterator from the runner loop would block the thread the consumer needs, so
        # this nested case runs on its own loop.
        return list(async_iter_to_sync_iter(source()))

    results: list[list[int]] = []
    thread = threading.Thread(target=lambda: results.append(run_async_in_sync(outer())), daemon=True)
    thread.start()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert results == [[1, 2]]
    assert source_loops[0] is not _get_runner_loop()
    assert source_loops[0].is_closed()
    # Work the source left behind is settled while the private loop is still alive, rather than
    # being abandoned to a loop that has already closed.
    assert background_cancelled.is_set()


def test_async_iter_to_sync_iter_disallows_running_loop_when_requested() -> None:
    async def source() -> AsyncIterator[int]:
        yield 1

    async def consume_in_async_context() -> None:
        with pytest.raises(RuntimeError, match="Cannot use the `sync` API in an `async` context"):
            list(async_iter_to_sync_iter(source(), allow_running_loop=False))

    asyncio.run(consume_in_async_context())


def test_async_iter_to_sync_iter_close_from_the_runner_thread_does_not_wedge_the_loop() -> None:
    """An abandoned iterator can be finalised on the runner thread by a garbage collection pass.

    The close runs the iterator's cleanup there, and waiting for the consumer to settle would
    block the only thread that can run the cancellation, taking the shared loop down for good.
    """

    async def infinite_source() -> AsyncIterator[int]:
        value = 0
        while True:
            await asyncio.sleep(0.001)
            yield value
            value += 1

    holder: dict[str, Any] = {}

    def run_in_thread() -> None:
        iterator = async_iter_to_sync_iter(infinite_source())
        assert next(iterator) == 0
        holder["iterator"] = iterator

        async def close_on_the_runner_loop() -> str:
            cast("Generator[int, None, None]", holder.pop("iterator")).close()
            return "closed"

        holder["result"] = run_async_in_sync(close_on_the_runner_loop())

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=10)

    assert not thread.is_alive(), "closing an abandoned iterator wedged the runner loop"
    assert holder["result"] == "closed"

    # The shared loop is still usable afterwards.
    follow_up: list[str] = []

    async def operation() -> str:
        return "still running"

    follow_up_thread = threading.Thread(target=lambda: follow_up.append(run_async_in_sync(operation())), daemon=True)
    follow_up_thread.start()
    follow_up_thread.join(timeout=5)

    assert not follow_up_thread.is_alive()
    assert follow_up == ["still running"]
