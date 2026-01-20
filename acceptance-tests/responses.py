"""Mock response generators for acceptance test scenarios."""

import time
import uuid
from collections.abc import AsyncIterator

from models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ChunkChoice,
    ChunkDelta,
    FunctionCall,
    ScenarioID,
    ToolCall,
    UsageInfo,
)


def generate_completion_id() -> str:
    """Generate a unique completion ID."""
    return f"chatcmpl-test-{uuid.uuid4().hex[:12]}"


def create_mock_response(
    request: ChatCompletionRequest,
    scenario: ScenarioID,
) -> ChatCompletionResponse:
    """Create a mock completion response based on the scenario."""
    completion_id = generate_completion_id()
    created = int(time.time())

    content = f"Mock response for scenario: {scenario.value}"
    finish_reason: str = "stop"
    tool_calls = None

    if scenario == ScenarioID.TOOL_CALLS and request.tools:
        finish_reason = "tool_calls"
        content = None
        tool_calls = [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=request.tools[0].function.name,
                    arguments='{"example": "arg"}',
                ),
            )
        ]

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,  # type: ignore[arg-type]
            )
        ],
        usage=UsageInfo(
            prompt_tokens=10,
            completion_tokens=15,
            total_tokens=25,
        ),
    )


async def create_streaming_response(
    request: ChatCompletionRequest,
    scenario: ScenarioID,
) -> AsyncIterator[ChatCompletionChunk]:
    """Create a streaming response as an async iterator of chunks."""
    completion_id = generate_completion_id()
    created = int(time.time())

    content = f"Mock streaming response for scenario: {scenario.value}"

    yield ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChunkDelta(role="assistant"),
                finish_reason=None,
            )
        ],
    )

    words = content.split()
    for word in words:
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChunkDelta(content=word + " "),
                    finish_reason=None,
                )
            ],
        )

    yield ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChunkDelta(),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=10,
            completion_tokens=len(words),
            total_tokens=10 + len(words),
        )
        if request.stream_options and request.stream_options.include_usage
        else None,
    )
