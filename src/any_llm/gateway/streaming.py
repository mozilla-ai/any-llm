"""Shared SSE streaming utilities for gateway routes."""

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from any_llm.gateway.log_config import logger
from any_llm.types.completion import CompletionUsage


@dataclass(frozen=True)
class StreamFormat:
    """SSE formatting configuration for a streaming protocol."""

    done_marker: str
    error_payload: str
    yield_done_on_error: bool


_OPENAI_ERROR = json.dumps({"error": {"message": "An error occurred during streaming", "type": "server_error"}})
_ANTHROPIC_ERROR = json.dumps(
    {"type": "error", "error": {"type": "api_error", "message": "An error occurred during streaming"}}
)

OPENAI_STREAM_FORMAT = StreamFormat(
    done_marker="data: [DONE]\n\n",
    error_payload=f"data: {_OPENAI_ERROR}\n\n",
    yield_done_on_error=True,
)

ANTHROPIC_STREAM_FORMAT = StreamFormat(
    done_marker="event: done\ndata: {}\n\n",
    error_payload=f"event: error\ndata: {_ANTHROPIC_ERROR}\n\n",
    yield_done_on_error=False,
)


def _merge_usage(current: CompletionUsage, update: CompletionUsage) -> CompletionUsage:
    """Merge usage data, keeping the last non-zero value for each field."""
    return CompletionUsage(
        prompt_tokens=update.prompt_tokens or current.prompt_tokens,
        completion_tokens=update.completion_tokens or current.completion_tokens,
        total_tokens=update.total_tokens or current.total_tokens,
    )


async def streaming_generator(
    stream: AsyncIterator[Any],
    format_chunk: Callable[[Any], str],
    extract_usage: Callable[[Any], CompletionUsage | None],
    fmt: StreamFormat,
    on_complete: Callable[[CompletionUsage], Awaitable[None]],
    on_error: Callable[[str], Awaitable[None]],
    label: str,
) -> AsyncIterator[str]:
    """Shared SSE streaming generator with usage tracking and error handling.

    Args:
        stream: Async iterator of chunks from the provider
        format_chunk: Formats a chunk into an SSE string
        extract_usage: Extracts usage from a chunk, or returns None if no usage present
        fmt: SSE format configuration (done marker, error payload, etc.)
        on_complete: Called with aggregated usage after successful streaming
        on_error: Called with error message on failure
        label: Identifier for error log messages (e.g., "openai:gpt-4")

    """
    usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    has_usage = False

    try:
        async for chunk in stream:
            chunk_usage = extract_usage(chunk)
            if chunk_usage:
                usage = _merge_usage(usage, chunk_usage)
                has_usage = True
            yield format_chunk(chunk)
        yield fmt.done_marker

        if has_usage:
            await on_complete(usage)
    except Exception as e:
        yield fmt.error_payload
        if fmt.yield_done_on_error:
            yield fmt.done_marker
        try:
            await on_error(str(e))
        except Exception as log_err:
            logger.error("Failed to log streaming error usage: %s", log_err)
        logger.error("Streaming error for %s: %s", label, e)
