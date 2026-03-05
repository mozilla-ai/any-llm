"""Tests for Anthropic provider native Messages API pass-through."""

from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from any_llm.providers.anthropic.base import BaseAnthropicProvider
from any_llm.types.messages import (
    MessageResponse,
    MessagesParams,
    MessageStreamEvent,
)


def _make_mock_message(**overrides: Any) -> Mock:
    """Create a mock Anthropic SDK Message object."""
    msg = Mock()
    msg.id = overrides.get("id", "msg_test123")
    msg.role = overrides.get("role", "assistant")
    msg.model = overrides.get("model", "claude-3-5-sonnet")
    msg.stop_reason = overrides.get("stop_reason", "end_turn")
    msg.type = "message"

    usage = Mock()
    usage.input_tokens = overrides.get("input_tokens", 10)
    usage.output_tokens = overrides.get("output_tokens", 5)
    usage.cache_creation_input_tokens = overrides.get("cache_creation_input_tokens", None)
    usage.cache_read_input_tokens = overrides.get("cache_read_input_tokens", None)
    msg.usage = usage

    msg.content = overrides.get("content", [])
    return msg


def _make_text_block(text: str = "Hello!") -> Mock:
    block = Mock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id: str = "toolu_123", name: str = "get_weather", tool_input: Any = None) -> Mock:
    block = Mock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = tool_input if tool_input is not None else {"city": "London"}
    return block


def _make_thinking_block(thinking: str = "Let me think...") -> Mock:
    block = Mock()
    block.type = "thinking"
    block.thinking = thinking
    return block


def test_convert_native_content_block_text() -> None:
    """Test converting an Anthropic text content block."""
    block = _make_text_block("Hello!")
    result = BaseAnthropicProvider._convert_native_content_block(block)
    assert result.type == "text"
    assert result.text == "Hello!"


def test_convert_native_content_block_tool_use() -> None:
    """Test converting an Anthropic tool_use content block."""
    block = _make_tool_use_block()
    result = BaseAnthropicProvider._convert_native_content_block(block)
    assert result.type == "tool_use"
    assert result.id == "toolu_123"
    assert result.name == "get_weather"
    assert result.input == {"city": "London"}


def test_convert_native_content_block_tool_use_non_dict_input() -> None:
    """Test tool_use block with non-dict input falls back to empty dict."""
    block = _make_tool_use_block(tool_input="not a dict")
    result = BaseAnthropicProvider._convert_native_content_block(block)
    assert result.input == {}


def test_convert_native_content_block_thinking() -> None:
    """Test converting an Anthropic thinking content block."""
    block = _make_thinking_block("Reasoning here")
    result = BaseAnthropicProvider._convert_native_content_block(block)
    assert result.type == "thinking"
    assert result.thinking == "Reasoning here"


def test_convert_native_content_block_unknown_type() -> None:
    """Test converting an unknown content block type returns minimal block."""
    block = Mock()
    block.type = "custom_block"
    result = BaseAnthropicProvider._convert_native_content_block(block)
    assert result.type == "custom_block"
    assert result.text is None
    assert result.id is None


def test_convert_native_message_to_response_text() -> None:
    """Test converting an Anthropic Message with text content."""
    msg = _make_mock_message(content=[_make_text_block("Hello!")])
    result = BaseAnthropicProvider._convert_native_message_to_response(msg)
    assert isinstance(result, MessageResponse)
    assert result.id == "msg_test123"
    assert result.role == "assistant"
    assert result.model == "claude-3-5-sonnet"
    assert result.stop_reason == "end_turn"
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "Hello!"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_convert_native_message_to_response_tool_use() -> None:
    """Test converting an Anthropic Message with tool_use content."""
    msg = _make_mock_message(
        content=[_make_tool_use_block()],
        stop_reason="tool_use",
    )
    result = BaseAnthropicProvider._convert_native_message_to_response(msg)
    assert result.stop_reason == "tool_use"
    assert len(result.content) == 1
    assert result.content[0].type == "tool_use"
    assert result.content[0].name == "get_weather"
    assert result.content[0].input == {"city": "London"}


def test_convert_native_message_to_response_tool_use_non_dict_input() -> None:
    """Test tool_use block with non-dict input in full message conversion."""
    msg = _make_mock_message(content=[_make_tool_use_block(tool_input="string_input")])
    result = BaseAnthropicProvider._convert_native_message_to_response(msg)
    assert result.content[0].input == {}


def test_convert_native_message_to_response_thinking() -> None:
    """Test converting an Anthropic Message with thinking content."""
    msg = _make_mock_message(
        content=[
            _make_thinking_block("Let me reason..."),
            _make_text_block("The answer is 42."),
        ]
    )
    result = BaseAnthropicProvider._convert_native_message_to_response(msg)
    assert len(result.content) == 2
    assert result.content[0].type == "thinking"
    assert result.content[0].thinking == "Let me reason..."
    assert result.content[1].type == "text"
    assert result.content[1].text == "The answer is 42."


def test_convert_native_message_to_response_cache_tokens() -> None:
    """Test that cache token fields are extracted from usage."""
    msg = _make_mock_message(
        content=[_make_text_block()],
        cache_creation_input_tokens=100,
        cache_read_input_tokens=50,
    )
    result = BaseAnthropicProvider._convert_native_message_to_response(msg)
    assert result.usage.cache_creation_input_tokens == 100
    assert result.usage.cache_read_input_tokens == 50


def test_convert_native_message_to_response_no_cache_tokens() -> None:
    """Test that cache tokens default to None when not present."""
    msg = _make_mock_message(content=[_make_text_block()])
    # Remove cache attributes to test getattr fallback
    del msg.usage.cache_creation_input_tokens
    del msg.usage.cache_read_input_tokens
    result = BaseAnthropicProvider._convert_native_message_to_response(msg)
    assert result.usage.cache_creation_input_tokens is None
    assert result.usage.cache_read_input_tokens is None


@pytest.mark.asyncio
async def test_amessages_non_streaming() -> None:
    """Test _amessages non-streaming calls client.messages.create."""
    mock_message = _make_mock_message(content=[_make_text_block("Hi!")])

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
    )
    result = await BaseAnthropicProvider._amessages(provider, params)
    assert isinstance(result, MessageResponse)
    assert result.content[0].text == "Hi!"

    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-3-5-sonnet"
    assert call_kwargs["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_amessages_non_streaming_with_all_params() -> None:
    """Test _amessages passes all optional params to API."""
    mock_message = _make_mock_message(content=[_make_text_block()])

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        system="Be helpful",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        stop_sequences=["END"],
        tools=[{"name": "fn", "description": "d", "input_schema": {}}],
        tool_choice={"type": "auto"},
        metadata={"user_id": "u1"},
        thinking={"type": "enabled", "budget_tokens": 8192},
    )
    result = await BaseAnthropicProvider._amessages(provider, params)
    assert isinstance(result, MessageResponse)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "Be helpful"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["top_p"] == 0.9
    assert call_kwargs["top_k"] == 40
    assert call_kwargs["stop_sequences"] == ["END"]
    assert call_kwargs["tools"] == [{"name": "fn", "description": "d", "input_schema": {}}]
    assert call_kwargs["tool_choice"] == {"type": "auto"}
    assert call_kwargs["metadata"] == {"user_id": "u1"}
    assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8192}


@pytest.mark.asyncio
async def test_amessages_none_params_not_included() -> None:
    """Test that None optional params are not passed to the API."""
    mock_message = _make_mock_message(content=[_make_text_block()])

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
    )
    await BaseAnthropicProvider._amessages(provider, params)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "system" not in call_kwargs
    assert "temperature" not in call_kwargs
    assert "tools" not in call_kwargs
    assert "thinking" not in call_kwargs


@pytest.mark.asyncio
async def test_amessages_streaming_delegates_to_stream_method() -> None:
    """Test _amessages with stream=True calls _stream_messages_async."""
    provider = Mock(spec=BaseAnthropicProvider)
    provider._stream_messages_async = Mock(return_value=AsyncMock())

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        stream=True,
    )
    await BaseAnthropicProvider._amessages(provider, params)
    provider._stream_messages_async.assert_called_once()


@pytest.mark.asyncio
async def test_stream_messages_async_emits_events() -> None:
    """Test _stream_messages_async converts Anthropic stream events to MessageStreamEvents."""
    from anthropic.types import (
        ContentBlockDeltaEvent,
        ContentBlockStartEvent,
        ContentBlockStopEvent,
        Message,
        MessageStartEvent,
        MessageStopEvent,
        TextBlock,
        TextDelta,
        Usage,
    )

    usage_start = Usage(input_tokens=10, output_tokens=0)
    msg = Message(
        id="msg_123",
        type="message",
        role="assistant",
        content=[],
        model="claude-3-5-sonnet",
        stop_reason=None,
        usage=usage_start,
    )
    stop_event = MessageStopEvent(type="message_stop")
    stop_event.message = Message(  # type: ignore[attr-defined]
        id="msg_123",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text="Hello!")],
        model="claude-3-5-sonnet",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
    )

    events_list: list[Any] = [
        MessageStartEvent(type="message_start", message=msg),
        ContentBlockStartEvent(type="content_block_start", index=0, content_block=TextBlock(type="text", text="")),
        ContentBlockDeltaEvent(type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="Hello!")),
        ContentBlockStopEvent(type="content_block_stop", index=0),
        stop_event,
    ]

    class MockStream:
        def __init__(self) -> None:
            self.events = iter(events_list)

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        def __aiter__(self) -> Self:
            return self

        async def __anext__(self) -> Any:
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration from None

    mock_client = Mock()
    mock_client.messages.stream = Mock(return_value=MockStream())

    provider = MagicMock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_content_block = BaseAnthropicProvider._convert_native_content_block

    collected: list[MessageStreamEvent] = []
    async for event in BaseAnthropicProvider._stream_messages_async(
        provider, model="claude-3-5-sonnet", messages=[], max_tokens=1024
    ):
        collected.append(event)

    types = [e.type for e in collected]
    assert "message_start" in types
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert "message_delta" in types
    assert "message_stop" in types

    msg_start = next(e for e in collected if e.type == "message_start")
    assert msg_start.message is not None
    assert msg_start.message.id == "msg_123"
    assert msg_start.message.usage.input_tokens == 10

    text_delta = next(e for e in collected if e.type == "content_block_delta")
    assert text_delta.delta is not None
    assert text_delta.delta["type"] == "text_delta"
    assert text_delta.delta["text"] == "Hello!"

    msg_delta = next(e for e in collected if e.type == "message_delta")
    assert msg_delta.delta is not None
    assert msg_delta.delta["stop_reason"] == "end_turn"
    assert msg_delta.usage is not None
    assert msg_delta.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_amessages_kwargs_passthrough() -> None:
    """Test that extra kwargs are passed through to the API call."""
    mock_message = _make_mock_message(content=[_make_text_block()])

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
    )
    await BaseAnthropicProvider._amessages(provider, params, custom_kwarg="value")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["custom_kwarg"] == "value"
