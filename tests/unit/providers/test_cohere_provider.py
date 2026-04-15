from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.cohere.utils import (
    _convert_response,
    _create_openai_chunk_from_cohere_chunk,
    _patch_messages,
)
from any_llm.types.completion import CompletionParams


def _mk_provider() -> Any:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    return CohereProvider(api_key="test-api-key")


def test_preprocess_response_format() -> None:
    provider = _mk_provider()

    class StructuredOutput(BaseModel):
        foo: str
        bar: int

    json_schema = {"type": "json_object", "schema": StructuredOutput.model_json_schema()}

    outp_basemodel = provider._preprocess_response_format(StructuredOutput)

    outp_dict = provider._preprocess_response_format(json_schema)

    assert isinstance(outp_basemodel, dict)
    assert isinstance(outp_dict, dict)

    assert outp_basemodel == outp_dict


@pytest.mark.asyncio
async def test_stream_and_response_format_combination_raises() -> None:
    provider = _mk_provider()

    with pytest.raises(UnsupportedParameterError):
        await provider._acompletion(
            CompletionParams(
                model_id="model-id",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                response_format={"type": "json_object"},
            )
        )


@pytest.mark.asyncio
async def test_parallel_tool_calls_raises() -> None:
    provider = _mk_provider()

    with pytest.raises(UnsupportedParameterError):
        await provider._acompletion(
            CompletionParams(
                model_id="model-id",
                messages=[{"role": "user", "content": "Hello"}],
                parallel_tool_calls=False,
            )
        )


def test_patch_messages_removes_name_from_tool_messages() -> None:
    """Test that _patch_messages removes 'name' field from tool messages."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "I'll check the weather for you.",
            "tool_calls": [{"id": "call_123", "function": {"name": "get_weather"}}],
        },
        {"role": "tool", "name": "get_weather", "content": "It's sunny", "tool_call_id": "call_123"},
        {"role": "assistant", "content": "The weather is sunny."},
    ]

    result = _patch_messages(messages)

    # Check that the tool message no longer has 'name' field
    tool_message = next(msg for msg in result if msg["role"] == "tool")
    assert "name" not in tool_message
    assert tool_message["content"] == "It's sunny"
    assert tool_message["tool_call_id"] == "call_123"

    # Check that other messages are unchanged
    user_message = next(msg for msg in result if msg["role"] == "user")
    assert user_message == {"role": "user", "content": "What's the weather?"}


def test_patch_messages_converts_assistant_content_to_tool_plan() -> None:
    """Test that _patch_messages converts assistant content to tool_plan when tool_calls present."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Calculate 2+2"},
        {
            "role": "assistant",
            "content": "I'll calculate that for you.",
            "tool_calls": [{"id": "call_456", "function": {"name": "calculator"}}],
        },
        {"role": "tool", "content": "4", "tool_call_id": "call_456"},
    ]

    result = _patch_messages(messages)

    # Check that assistant message with tool_calls has content moved to tool_plan
    assistant_message = next(msg for msg in result if msg["role"] == "assistant" and msg.get("tool_calls"))
    assert "content" not in assistant_message
    assert assistant_message["tool_plan"] == "I'll calculate that for you."
    assert assistant_message["tool_calls"] == [{"id": "call_456", "function": {"name": "calculator"}}]


def test_patch_messages_leaves_regular_assistant_messages_unchanged() -> None:
    """Test that _patch_messages doesn't modify assistant messages without tool_calls."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
        {"role": "user", "content": "Thanks"},
    ]

    result = _patch_messages(messages)

    # Messages should be unchanged
    assert result == messages

    # Verify assistant message still has content
    assistant_message = next(msg for msg in result if msg["role"] == "assistant")
    assert assistant_message["content"] == "Hello! How can I help you?"
    assert "tool_plan" not in assistant_message


def test_patch_messages_with_invalid_tool_sequence_raises_error() -> None:
    """Test that an invalid tool message sequence raises a ValueError."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather?"},
        {"role": "tool", "name": "get_weather", "content": "It's sunny", "tool_call_id": "call_123"},
    ]
    with pytest.raises(ValueError, match=r"A tool message must be preceded by an assistant message with tool_calls."):
        _patch_messages(messages)


def test_patch_messages_with_parallel_tool_messages() -> None:
    """Test that multiple consecutive tool messages after an assistant message are valid."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Get weather for Paris and London."},
        {
            "role": "assistant",
            "content": "I'll check both cities.",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"location":"Paris"}'}},
                {"id": "call_2", "function": {"name": "get_weather", "arguments": '{"location":"London"}'}},
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "Sunny in Paris", "tool_call_id": "call_1"},
        {"role": "tool", "name": "get_weather", "content": "Rainy in London", "tool_call_id": "call_2"},
    ]

    result = _patch_messages(messages)

    tool_messages = [msg for msg in result if msg["role"] == "tool"]
    assert len(tool_messages) == 2
    for msg in tool_messages:
        assert "name" not in msg


def test_patch_messages_tool_after_tool_without_assistant_raises() -> None:
    """Test that a tool message group not preceded by an assistant message raises."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": "result1", "tool_call_id": "call_1"},
        {"role": "tool", "content": "result2", "tool_call_id": "call_2"},
    ]
    with pytest.raises(ValueError, match=r"A tool message must be preceded by an assistant message with tool_calls."):
        _patch_messages(messages)


def test_preprocess_response_format_dataclass() -> None:
    from dataclasses import dataclass

    provider = _mk_provider()

    @dataclass
    class StructuredOutputDC:
        foo: str
        bar: int

    result = provider._preprocess_response_format(StructuredOutputDC)

    assert isinstance(result, dict)
    assert result["type"] == "json_object"
    assert "properties" in result["schema"]
    assert "foo" in result["schema"]["properties"]
    assert "bar" in result["schema"]["properties"]


def test_preprocess_response_format_unsupported_raises() -> None:
    provider = _mk_provider()

    with pytest.raises(ValueError, match="Unsupported response_format"):
        provider._preprocess_response_format("invalid")


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", ["auto", "none"])
async def test_reasoning_effort_filtered_out(reasoning_effort: str) -> None:
    """Test that reasoning_effort 'auto' and 'none' are filtered from Cohere API calls."""
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    with patch("any_llm.providers.cohere.cohere.cohere") as mock_cohere:
        mock_client = Mock()
        mock_cohere.AsyncClientV2.return_value = mock_client
        mock_client.chat = AsyncMock(return_value=Mock())

        with patch("any_llm.providers.cohere.cohere._convert_response", return_value=Mock()):
            provider = CohereProvider(api_key="test-api-key")
            await provider._acompletion(
                CompletionParams(
                    model_id="command-r-plus",
                    messages=[{"role": "user", "content": "Hello"}],
                    reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
                ),
            )

            call_kwargs = mock_client.chat.call_args[1]
            assert "reasoning_effort" not in call_kwargs


def _mock_tool_call(tool_id: str, name: str, arguments: str) -> Mock:
    """Create a mock Cohere ToolCallV2."""
    tc = Mock()
    tc.id = tool_id
    tc.function = Mock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _mock_v2chat_response(tool_calls: list[Mock], tool_plan: str = "I'll help.") -> Mock:
    """Create a mock V2ChatResponse with multiple tool calls."""
    response = Mock()
    response.finish_reason = "TOOL_CALL"
    response.id = "resp-123"
    response.created = 0
    response.message = Mock()
    response.message.tool_calls = tool_calls
    response.message.tool_plan = tool_plan
    response.usage = Mock()
    response.usage.tokens = Mock()
    response.usage.tokens.input_tokens = 10
    response.usage.tokens.output_tokens = 20
    return response


def test_convert_response_multiple_tool_calls() -> None:
    """Non-streaming: all tool calls are converted, not just the first."""
    tc1 = _mock_tool_call("call_1", "get_weather", '{"city":"NYC"}')
    tc2 = _mock_tool_call("call_2", "get_time", '{"tz":"EST"}')
    tc3 = _mock_tool_call("call_3", "get_news", '{"topic":"tech"}')

    result = _convert_response(_mock_v2chat_response([tc1, tc2, tc3]), model="command-r-plus")

    assert len(result.choices) == 1
    assert result.choices[0].finish_reason == "tool_calls"
    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 3
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].function.name == "get_weather"  # type: ignore[union-attr]
    assert tool_calls[0].function.arguments == '{"city":"NYC"}'  # type: ignore[union-attr]
    assert tool_calls[1].id == "call_2"
    assert tool_calls[1].function.name == "get_time"  # type: ignore[union-attr]
    assert tool_calls[2].id == "call_3"
    assert tool_calls[2].function.name == "get_news"  # type: ignore[union-attr]


def test_convert_response_single_tool_call() -> None:
    """Non-streaming: single tool call still works correctly."""
    tc = _mock_tool_call("call_1", "get_weather", '{"city":"NYC"}')

    result = _convert_response(_mock_v2chat_response([tc]), model="command-r-plus")

    assert len(result.choices[0].message.tool_calls) == 1  # type: ignore[arg-type]
    assert result.choices[0].message.tool_calls[0].id == "call_1"  # type: ignore[index]


def test_streaming_tool_call_start_uses_chunk_index() -> None:
    """Streaming: tool-call-start uses chunk.index, not hardcoded 0."""
    chunk = Mock()
    chunk.type = "tool-call-start"
    chunk.index = 2
    chunk.delta = Mock()
    chunk.delta.message = Mock()
    chunk.delta.message.tool_calls = Mock()
    chunk.delta.message.tool_calls.id = "call_abc"
    chunk.delta.message.tool_calls.function = Mock()
    chunk.delta.message.tool_calls.function.name = "get_weather"

    result = _create_openai_chunk_from_cohere_chunk(chunk)

    tool_calls = result.choices[0].delta.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].index == 2
    assert tool_calls[0].id == "call_abc"
    assert tool_calls[0].function.name == "get_weather"  # type: ignore[union-attr]


def test_streaming_tool_call_delta_uses_chunk_index() -> None:
    """Streaming: tool-call-delta uses chunk.index, not hardcoded 0."""
    chunk = Mock()
    chunk.type = "tool-call-delta"
    chunk.index = 3
    chunk.delta = Mock()
    chunk.delta.message = Mock()
    chunk.delta.message.tool_calls = Mock()
    chunk.delta.message.tool_calls.function = Mock()
    chunk.delta.message.tool_calls.function.arguments = '{"partial'

    result = _create_openai_chunk_from_cohere_chunk(chunk)

    tool_calls = result.choices[0].delta.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].index == 3
    assert tool_calls[0].function.arguments == '{"partial'  # type: ignore[union-attr]


def test_streaming_tool_call_index_defaults_to_zero_when_none() -> None:
    """Streaming: falls back to index 0 when chunk.index is None."""
    chunk = Mock()
    chunk.type = "tool-call-start"
    chunk.index = None
    chunk.delta = Mock()
    chunk.delta.message = Mock()
    chunk.delta.message.tool_calls = Mock()
    chunk.delta.message.tool_calls.id = "call_xyz"
    chunk.delta.message.tool_calls.function = Mock()
    chunk.delta.message.tool_calls.function.name = "search"

    result = _create_openai_chunk_from_cohere_chunk(chunk)

    assert result.choices[0].delta.tool_calls[0].index == 0  # type: ignore[index]
