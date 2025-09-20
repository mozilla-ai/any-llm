import asyncio
import pytest
from any_llm.utils.aio import run_async_in_sync


def test_run_async_in_sync_fails_with_background_task_state():
    task_completed = {"value": False}

    async def operation_with_critical_background_task():
        """Simulates an operation where a background task MUST complete for success."""

        async def critical_background_work():
            await asyncio.sleep(0.02)
            task_completed["value"] = True

        asyncio.create_task(critical_background_work())
        return "operation_started"

    async def test_in_streamlit_context():
        task_completed["value"] = False

        # This triggers the threading path in run_async_in_sync
        result = run_async_in_sync(operation_with_critical_background_task())
        assert result == "operation_started"

        await asyncio.sleep(0.05)

        assert task_completed["value"] is True, (
            "FAILURE: Background task was orphaned when thread's event loop closed. "
            "This demonstrates the bug in run_async_in_sync!"
        )

    asyncio.run(test_in_streamlit_context())
