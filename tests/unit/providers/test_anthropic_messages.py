"""Tests for Anthropic provider native Messages API pass-through."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage
from pydantic import BaseModel

from any_llm.providers.anthropic.anthropic import AnthropicProvider
from any_llm.providers.anthropic.base import BaseAnthropicProvider
from any_llm.types.messages import (
    CompactionDelta,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageResponse,
    MessagesParams,
    MessageStartEvent,
    MessageStopEvent,
)


class _ContextEditModel(BaseModel):
    type: str


@dataclass
class _ContextEditDataclass:
    type: str


@dataclass
class _UnknownContextEdit:
    value: str


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
async def test_amessages_context_compaction_uses_beta_resource_and_preserves_response() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"request-id": "req_test"},
            json={
                "id": "msg_compaction",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "stop_reason": "compaction",
                "stop_sequence": None,
                "content": [{"type": "compaction", "content": "Conversation summary"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "iterations": [
                        {
                            "type": "compaction",
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        }
                    ],
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", http_client=http_client)
    context_management = {"edits": [{"type": "compact_20260112"}]}
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        context_management=context_management,
    )

    try:
        result = await provider._amessages(params)
    finally:
        await http_client.aclose()

    assert isinstance(result, MessageResponse)
    assert result.stop_reason == "compaction"
    assert result.content[0].type == "compaction"
    assert result.content[0].content == "Conversation summary"
    assert result.usage.iterations is not None
    assert result.usage.iterations[0].type == "compaction"

    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/v1/messages"
    assert request.url.query == b"beta=true"
    assert request.headers["anthropic-beta"] == "compact-2026-01-12"
    request_body = json.loads(request.content)
    assert request_body["context_management"] == context_management


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("context_management", "betas", "expected_betas"),
    [
        (
            {"edits": [{"type": "compact_20260112"}]},
            ["custom-beta", "compact-2026-01-12"],
            "custom-beta,compact-2026-01-12",
        ),
        (
            {"edits": [{"type": "clear_tool_uses_20250919"}]},
            None,
            "context-management-2025-06-27",
        ),
        (
            {"edits": [_ContextEditModel(type="compact_20260112")]},
            None,
            "compact-2026-01-12",
        ),
        (
            {"edits": [_ContextEditDataclass(type="compact_20260112")]},
            None,
            "compact-2026-01-12",
        ),
        (
            {"edits": [_UnknownContextEdit(value="custom")]},
            ["custom-beta"],
            "custom-beta",
        ),
        (None, ["custom-beta"], "custom-beta"),
    ],
)
async def test_amessages_selects_betas_for_context_management(
    context_management: dict[str, Any] | None, betas: list[str] | None, expected_betas: str
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"request-id": "req_test"},
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "content": [{"type": "text", "text": "Done"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", http_client=http_client)
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        context_management=context_management,
        betas=betas,
    )

    try:
        await provider._amessages(params)
    finally:
        await http_client.aclose()

    assert requests[0].headers["anthropic-beta"] == expected_betas


@pytest.mark.asyncio
async def test_amessages_streams_beta_compaction_events() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.query == b"beta=true"
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_compaction",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-opus-5",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "content": [],
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "compaction", "content": None},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "compaction_delta", "content": "Conversation summary"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "compaction", "stop_sequence": None},
                    "usage": {"output_tokens": 5},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", http_client=http_client)
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        stream=True,
        context_management={"edits": [{"type": "compact_20260112"}]},
    )

    try:
        stream = await provider._amessages(params)
        assert isinstance(stream, AsyncIterator)
        collected = [event async for event in stream]
    finally:
        await http_client.aclose()

    assert [event.type for event in collected] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    content_start = collected[1]
    assert isinstance(content_start, ContentBlockStartEvent)
    assert content_start.content_block.type == "compaction"

    content_delta = collected[2]
    assert isinstance(content_delta, ContentBlockDeltaEvent)
    assert isinstance(content_delta.delta, CompactionDelta)
    assert content_delta.delta.type == "compaction_delta"
    assert content_delta.delta.content == "Conversation summary"

    message_delta = collected[4]
    assert isinstance(message_delta, MessageDeltaEvent)
    assert message_delta.delta.stop_reason == "compaction"


@pytest.mark.asyncio
async def test_amessages_non_streaming_with_all_params() -> None:
    """Test _amessages passes all optional params to API."""
    mock_message = _make_message(content=[TextBlock(type="text", text="Hello!")])

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
async def test_amessages_output_format_uses_native_parse() -> None:
    """With output_format set, the native path returns messages.parse output unchanged."""
    from anthropic.types.parsed_message import ParsedMessage, ParsedTextBlock
    from pydantic import BaseModel

    class City(BaseModel):
        city_name: str

    parsed_message = ParsedMessage[City](
        id="msg_parse",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet",
        stop_reason="end_turn",
        stop_sequence=None,
        content=[
            ParsedTextBlock[City](
                type="text",
                text='{"city_name": "Paris"}',
                citations=None,
                parsed_output=City(city_name="Paris"),
            )
        ],
        usage=_make_usage(),
    )

    mock_client = Mock()
    mock_client.messages.parse = AsyncMock(return_value=parsed_message)
    mock_client.messages.create = AsyncMock()

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Capital of France?"}],
        max_tokens=1024,
        output_format=City,
    )
    result = await BaseAnthropicProvider._amessages(provider, params)

    # The SDK's ParsedMessage is returned as-is, no conversion or re-validation.
    assert isinstance(result, ParsedMessage)
    assert result is parsed_message
    assert result.parsed_output == City(city_name="Paris")
    mock_client.messages.create.assert_not_called()

    # output_format is passed to parse as its dedicated kwarg; other params still flow through.
    call_kwargs = mock_client.messages.parse.call_args.kwargs
    assert call_kwargs["output_format"] is City


@pytest.mark.asyncio
async def test_amessages_output_config_dict_passes_through_to_create() -> None:
    """A raw output_config dict goes to native messages.create(output_config=...), not parse."""
    output_config = {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    mock_message = _make_message(content=[TextBlock(type="text", text='{"city_name": "Paris"}')])

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)
    mock_client.messages.parse = AsyncMock()

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Capital of France?"}],
        max_tokens=1024,
        output_format=output_config,
    )
    result = await BaseAnthropicProvider._amessages(provider, params)

    # The raw-dict path returns a MessageResponse (the base layer builds the ParsedMessage).
    assert isinstance(result, MessageResponse)
    mock_client.messages.parse.assert_not_called()
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["output_config"] == output_config
    assert "output_format" not in call_kwargs
    assert call_kwargs["model"] == "claude-3-5-sonnet"
    assert call_kwargs["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_amessages_cache_control_passthrough() -> None:
    """Test that cache_control is passed through to the API call."""
    mock_message = _make_message(content=[TextBlock(type="text", text="Hello!")])

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response

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
    """Test _stream_messages_async yields SDK event types directly."""
    from anthropic.types import (
        ContentBlockDeltaEvent as SDKContentBlockDeltaEvent,
    )
    from anthropic.types import (
        ContentBlockStartEvent as SDKContentBlockStartEvent,
    )
    from anthropic.types import (
        ContentBlockStopEvent as SDKContentBlockStopEvent,
    )
    from anthropic.types import (
        MessageDeltaUsage,
        TextDelta,
    )
    from anthropic.types import (
        MessageStartEvent as SDKMessageStartEvent,
    )
    from anthropic.types import (
        MessageStopEvent as SDKMessageStopEvent,
    )
    from anthropic.types.raw_message_delta_event import Delta as SDKDelta

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

    from anthropic.types import RawMessageDeltaEvent

    events_list: list[Any] = [
        SDKMessageStartEvent(type="message_start", message=msg),
        SDKContentBlockStartEvent(type="content_block_start", index=0, content_block=TextBlock(type="text", text="")),
        SDKContentBlockDeltaEvent(
            type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="Hello!")
        ),
        SDKContentBlockStopEvent(type="content_block_stop", index=0),
        RawMessageDeltaEvent(
            type="message_delta",
            delta=SDKDelta(stop_reason="end_turn"),
            usage=MessageDeltaUsage(output_tokens=5),
        ),
        SDKMessageStopEvent(type="message_stop"),
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

    collected: list[
        MessageStartEvent
        | MessageDeltaEvent
        | MessageStopEvent
        | ContentBlockStartEvent
        | ContentBlockStopEvent
        | ContentBlockDeltaEvent
    ] = []
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

    msg_start = next(e for e in collected if isinstance(e, MessageStartEvent))
    assert msg_start.message.id == "msg_123"
    assert msg_start.message.usage.input_tokens == 10

    text_delta = next(e for e in collected if isinstance(e, ContentBlockDeltaEvent))
    assert text_delta.delta.type == "text_delta"
    assert text_delta.delta.text == "Hello!"

    msg_delta = next(e for e in collected if isinstance(e, MessageDeltaEvent))
    assert msg_delta.delta.stop_reason == "end_turn"
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
