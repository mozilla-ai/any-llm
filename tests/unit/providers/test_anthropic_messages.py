"""Tests for Anthropic provider native Messages API pass-through."""

from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage

from any_llm.providers.anthropic.base import BaseAnthropicProvider
from any_llm.types.messages import (
    MessageResponse,
    MessagesParams,
    MessageStreamEvent,
)
from any_llm.types.messages import (
    ThinkingBlock as AnyLLMThinkingBlock,
)


def _make_usage(**overrides: Any) -> Usage:
    defaults: dict[str, Any] = {"input_tokens": 10, "output_tokens": 5}
    defaults.update(overrides)
    return Usage(**defaults)


def _make_message(**overrides: Any) -> Message:
    defaults: dict[str, Any] = {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet",
        "stop_reason": "end_turn",
        "content": [],
        "usage": _make_usage(),
    }
    defaults.update(overrides)
    return Message(**defaults)


def test_convert_native_content_block_text() -> None:
    """Test converting an Anthropic text content block."""
    block = TextBlock(type="text", text="Hello!")
    result = BaseAnthropicProvider._convert_native_content_block(block)
    assert result.type == "text"
    assert result.text == "Hello!"


def test_convert_native_content_block_tool_use() -> None:
    """Test converting an Anthropic tool_use content block."""
    block = ToolUseBlock(type="tool_use", id="toolu_123", name="get_weather", input={"city": "London"})
    result = BaseAnthropicProvider._convert_native_content_block(block)
    assert result.type == "tool_use"
    assert result.id == "toolu_123"
    assert result.name == "get_weather"
    assert result.input == {"city": "London"}


def test_convert_native_content_block_thinking() -> None:
    """Test converting an Anthropic thinking content block wraps as our ThinkingBlock."""
    block = ThinkingBlock(type="thinking", thinking="Reasoning here", signature="sig123")
    result = BaseAnthropicProvider._convert_native_content_block(block)
    assert isinstance(result, AnyLLMThinkingBlock)
    assert result.type == "thinking"
    assert result.thinking == "Reasoning here"
    assert result.signature == "sig123"


def test_convert_native_content_block_thinking_preserves_empty_signature() -> None:
    """Test that ThinkingBlock with empty signature is preserved."""
    block = ThinkingBlock(type="thinking", thinking="test", signature="")
    result = BaseAnthropicProvider._convert_native_content_block(block)
    assert isinstance(result, AnyLLMThinkingBlock)
    assert result.signature == ""


def test_convert_native_content_block_passes_through_other_types() -> None:
    """Test that non-ThinkingBlock content blocks are passed through as-is."""
    block = TextBlock(type="text", text="Hi!")
    result = BaseAnthropicProvider._convert_native_content_block(block)
    assert result is block


def test_convert_native_message_to_response_text() -> None:
    """Test converting an Anthropic Message with text content."""
    msg = _make_message(content=[TextBlock(type="text", text="Hello!")])
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
    msg = _make_message(
        content=[ToolUseBlock(type="tool_use", id="toolu_123", name="get_weather", input={"city": "London"})],
        stop_reason="tool_use",
    )
    result = BaseAnthropicProvider._convert_native_message_to_response(msg)
    assert result.stop_reason == "tool_use"
    assert len(result.content) == 1
    assert result.content[0].type == "tool_use"
    assert result.content[0].name == "get_weather"
    assert result.content[0].input == {"city": "London"}


def test_convert_native_message_to_response_thinking() -> None:
    """Test converting an Anthropic Message with thinking content."""
    msg = _make_message(
        content=[
            ThinkingBlock(type="thinking", thinking="Let me reason...", signature="sig"),
            TextBlock(type="text", text="The answer is 42."),
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
    msg = _make_message(
        content=[TextBlock(type="text", text="Hello!")],
        usage=_make_usage(cache_creation_input_tokens=100, cache_read_input_tokens=50),
    )
    result = BaseAnthropicProvider._convert_native_message_to_response(msg)
    assert result.usage.cache_creation_input_tokens == 100
    assert result.usage.cache_read_input_tokens == 50


def test_convert_native_message_to_response_no_cache_tokens() -> None:
    """Test that cache tokens default to None when not set."""
    msg = _make_message(content=[TextBlock(type="text", text="Hello!")])
    result = BaseAnthropicProvider._convert_native_message_to_response(msg)
    assert result.usage.cache_creation_input_tokens is None
    assert result.usage.cache_read_input_tokens is None


@pytest.mark.asyncio
async def test_amessages_non_streaming() -> None:
    """Test _amessages non-streaming calls client.messages.create."""
    mock_message = _make_message(content=[TextBlock(type="text", text="Hi!")])

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response
    provider._convert_native_content_block = BaseAnthropicProvider._convert_native_content_block

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
    )
    result = await BaseAnthropicProvider._amessages(provider, params)
    assert isinstance(result, MessageResponse)
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "Hi!"

    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-3-5-sonnet"
    assert call_kwargs["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_amessages_non_streaming_with_all_params() -> None:
    """Test _amessages passes all optional params to API."""
    mock_message = _make_message(content=[TextBlock(type="text", text="Hello!")])

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response
    provider._convert_native_content_block = BaseAnthropicProvider._convert_native_content_block

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
async def test_amessages_cache_control_passthrough() -> None:
    """Test that cache_control is passed through to the API call."""
    mock_message = _make_message(content=[TextBlock(type="text", text="Hello!")])

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response
    provider._convert_native_content_block = BaseAnthropicProvider._convert_native_content_block

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        cache_control={"type": "ephemeral"},
    )
    await BaseAnthropicProvider._amessages(provider, params)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_amessages_none_params_not_included() -> None:
    """Test that None optional params are not passed to the API."""
    mock_message = _make_message(content=[TextBlock(type="text", text="Hello!")])

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response
    provider._convert_native_content_block = BaseAnthropicProvider._convert_native_content_block

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
    assert "cache_control" not in call_kwargs


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
        MessageStartEvent,
        MessageStopEvent,
        TextDelta,
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
    mock_message = _make_message(content=[TextBlock(type="text", text="Hello!")])

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response
    provider._convert_native_content_block = BaseAnthropicProvider._convert_native_content_block

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
    )
    await BaseAnthropicProvider._amessages(provider, params, custom_kwarg="value")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["custom_kwarg"] == "value"


@pytest.mark.asyncio
async def test_amessages_system_list_form() -> None:
    """Test that system param accepts list of content blocks."""
    mock_message = _make_message(content=[TextBlock(type="text", text="Hello!")])

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response
    provider._convert_native_content_block = BaseAnthropicProvider._convert_native_content_block

    system_blocks = [
        {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}},
    ]
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        system=system_blocks,
    )
    await BaseAnthropicProvider._amessages(provider, params)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == system_blocks
