from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from together.types import ChatCompletionChunk as TogetherChatCompletionChunk

from any_llm.providers.together.utils import _create_openai_chunk_from_together_chunk
from any_llm.types.completion import ChatCompletionChunk, CompletionParams


def make_together_chunk(
    content: str | None = None,
    role: str = "assistant",
    reasoning: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
) -> Mock:
    """Create a mock TogetherChatCompletionChunk for testing."""
    delta_mock = Mock()
    delta_mock.content = content
    delta_mock.role = role
    delta_mock.reasoning = reasoning
    delta_mock.tool_calls = tool_calls

    choice_mock = Mock()
    choice_mock.delta = delta_mock
    choice_mock.index = 0
    choice_mock.finish_reason = finish_reason

    usage_mock = None
    if usage:
        usage_mock = Mock()
        usage_mock.prompt_tokens = usage.get("prompt_tokens", 0)
        usage_mock.completion_tokens = usage.get("completion_tokens", 0)
        usage_mock.total_tokens = usage.get("total_tokens", 0)

    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = [choice_mock]
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = usage_mock

    return together_chunk


def test_create_openai_chunk_handles_empty_choices() -> None:
    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = None
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = None

    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert isinstance(result, ChatCompletionChunk)
    assert result.choices == []
    assert result.id == "test-id"
    assert result.model == "test-model"

    together_chunk.choices = []
    result = _create_openai_chunk_from_together_chunk(together_chunk)
    assert result.choices == []


def test_create_openai_chunk_handles_missing_delta() -> None:
    """Test that the function handles choices with None delta gracefully."""
    choice_mock = Mock()
    choice_mock.delta = None
    choice_mock.index = 0
    choice_mock.finish_reason = "stop"

    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = [choice_mock]
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = None

    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert len(result.choices) == 1
    assert result.choices[0].delta.content is None
    assert result.choices[0].delta.role is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", ["auto", "none"])
async def test_reasoning_effort_filtered_out(reasoning_effort: str) -> None:
    """Test that reasoning_effort 'auto' and 'none' are filtered from Together API calls."""
    from any_llm.providers.together.together import TogetherProvider

    with (
        patch("together.AsyncTogether") as mock_together,
        patch.object(TogetherProvider, "_convert_completion_response", return_value=Mock()),
    ):
        mock_client = Mock()
        mock_together.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=Mock())

        provider = TogetherProvider(api_key="test-api-key")
        await provider._acompletion(
            CompletionParams(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            ),
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "reasoning_effort" not in call_kwargs


def test_create_openai_chunk_with_tool_calls_dict_format() -> None:
    """Test handling tool calls in dict format."""
    tool_call_dict = {
        "id": "call-123",
        "index": 0,
        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
    }

    together_chunk = make_together_chunk(tool_calls=[tool_call_dict], finish_reason="tool_calls")
    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert len(result.choices) == 1
    assert result.choices[0].delta.tool_calls is not None
    assert len(result.choices[0].delta.tool_calls) == 1
    tool_call = result.choices[0].delta.tool_calls[0]
    assert tool_call.id == "call-123"
    assert tool_call.index == 0
    assert tool_call.function is not None
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"location": "Paris"}'


def test_create_openai_chunk_with_tool_calls_object_format() -> None:
    """Test handling tool calls in object format."""
    func_mock = Mock()
    func_mock.name = "search"
    func_mock.arguments = '{"query": "test"}'

    tool_call_mock = Mock()
    tool_call_mock.id = "call-456"
    tool_call_mock.index = 0
    tool_call_mock.function = func_mock

    together_chunk = make_together_chunk(tool_calls=[tool_call_mock], finish_reason="tool_calls")
    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert result.choices[0].delta.tool_calls is not None
    tool_call = result.choices[0].delta.tool_calls[0]
    assert tool_call.id == "call-456"
    assert tool_call.function is not None
    assert tool_call.function.name == "search"
    assert tool_call.function.arguments == '{"query": "test"}'


def test_create_openai_chunk_with_tool_calls_missing_id() -> None:
    """Test handling tool calls without id generates uuid."""
    tool_call_dict = {
        "index": 0,
        "function": {"name": "func", "arguments": "{}"},
    }

    together_chunk = make_together_chunk(tool_calls=[tool_call_dict], finish_reason="tool_calls")
    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert result.choices[0].delta.tool_calls is not None
    tool_call = result.choices[0].delta.tool_calls[0]
    assert tool_call.id is not None
    assert len(tool_call.id) > 0


def test_create_openai_chunk_with_tool_calls_object_missing_function() -> None:
    """Test handling tool calls object without function attribute."""
    tool_call_mock = Mock()
    tool_call_mock.id = None
    tool_call_mock.index = 0
    tool_call_mock.function = None

    together_chunk = make_together_chunk(tool_calls=[tool_call_mock], finish_reason="tool_calls")
    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert result.choices[0].delta.tool_calls is not None
    tool_call = result.choices[0].delta.tool_calls[0]
    assert tool_call.function is not None
    assert tool_call.function.name == ""
    assert tool_call.function.arguments == ""


def test_create_openai_chunk_with_reasoning() -> None:
    """Test handling reasoning content."""
    together_chunk = make_together_chunk(content="The answer is 42", reasoning="Let me think about this...")
    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert result.choices[0].delta.content == "The answer is 42"
    assert result.choices[0].delta.reasoning is not None
    assert result.choices[0].delta.reasoning.content == "Let me think about this..."


def test_create_openai_chunk_with_usage() -> None:
    """Test handling usage metadata."""
    together_chunk = make_together_chunk(
        content="Hello",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15
