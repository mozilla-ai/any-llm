"""Tests for the shared streaming_generator utility."""

import pytest

from any_llm.gateway.streaming import (
    ANTHROPIC_STREAM_FORMAT,
    OPENAI_STREAM_FORMAT,
    streaming_generator,
)
from any_llm.types.completion import CompletionUsage


def _format_chunk(chunk: str) -> str:
    return f"data: {chunk}\n\n"


def _extract_usage(chunk: str) -> CompletionUsage | None:
    if chunk == "usage":
        return CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return None


async def _items(*values: str):
    for v in values:
        yield v


@pytest.mark.asyncio
async def test_streaming_generator_success_with_usage() -> None:
    """Test successful streaming with usage tracking."""
    completed_usage: list[CompletionUsage] = []

    async def on_complete(usage: CompletionUsage) -> None:
        completed_usage.append(usage)

    async def on_error(error: str) -> None:
        pytest.fail("on_error should not be called")

    events: list[str] = []
    async for event in streaming_generator(
        stream=_items("hello", "usage"),
        format_chunk=_format_chunk,
        extract_usage=_extract_usage,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
    ):
        events.append(event)

    assert events == ["data: hello\n\n", "data: usage\n\n", "data: [DONE]\n\n"]
    assert len(completed_usage) == 1
    assert completed_usage[0].prompt_tokens == 10
    assert completed_usage[0].completion_tokens == 5


@pytest.mark.asyncio
async def test_streaming_generator_no_usage_skips_on_complete() -> None:
    """Test that on_complete is not called when no usage data is received."""
    completed = False

    async def on_complete(usage: CompletionUsage) -> None:
        nonlocal completed
        completed = True

    async def on_error(error: str) -> None:
        pytest.fail("on_error should not be called")

    events: list[str] = []
    async for event in streaming_generator(
        stream=_items("hello"),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
    ):
        events.append(event)

    assert events == ["data: hello\n\n", "data: [DONE]\n\n"]
    assert not completed


@pytest.mark.asyncio
async def test_streaming_generator_error_openai_format() -> None:
    """Test error handling emits OpenAI-style error and [DONE]."""
    error_logged: list[str] = []

    async def on_complete(usage: CompletionUsage) -> None:
        pytest.fail("on_complete should not be called on error")

    async def on_error(error: str) -> None:
        error_logged.append(error)

    async def _failing_stream():
        yield "hello"
        raise RuntimeError("provider crashed")

    events: list[str] = []
    async for event in streaming_generator(
        stream=_failing_stream(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
    ):
        events.append(event)

    assert events[0] == "data: hello\n\n"
    assert "server_error" in events[1]
    assert events[2] == "data: [DONE]\n\n"
    assert error_logged == ["provider crashed"]


@pytest.mark.asyncio
async def test_streaming_generator_error_anthropic_format() -> None:
    """Test error handling emits Anthropic-style error without done marker."""
    error_logged: list[str] = []

    async def on_complete(usage: CompletionUsage) -> None:
        pytest.fail("on_complete should not be called on error")

    async def on_error(error: str) -> None:
        error_logged.append(error)

    async def _failing_stream():
        raise RuntimeError("provider crashed")
        yield  # pragma: no cover

    events: list[str] = []
    async for event in streaming_generator(
        stream=_failing_stream(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=ANTHROPIC_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
    ):
        events.append(event)

    assert len(events) == 1
    assert "api_error" in events[0]
    assert events[0].startswith("event: error\n")
    assert error_logged == ["provider crashed"]


@pytest.mark.asyncio
async def test_streaming_generator_error_logging_failure_is_swallowed() -> None:
    """Test that failures in on_error don't propagate to the caller."""

    async def on_complete(usage: CompletionUsage) -> None:
        pytest.fail("on_complete should not be called on error")

    async def on_error(error: str) -> None:
        raise RuntimeError("logging failed too")

    async def _failing_stream():
        raise RuntimeError("provider crashed")
        yield  # pragma: no cover

    events: list[str] = []
    async for event in streaming_generator(
        stream=_failing_stream(),
        format_chunk=_format_chunk,
        extract_usage=lambda _: None,
        fmt=OPENAI_STREAM_FORMAT,
        on_complete=on_complete,
        on_error=on_error,
        label="test:model",
    ):
        events.append(event)

    assert "server_error" in events[0]
    assert events[1] == "data: [DONE]\n\n"
