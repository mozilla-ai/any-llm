import asyncio

from any_llm.utils.aio import run_async_in_sync


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
