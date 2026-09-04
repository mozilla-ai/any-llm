"""Tests for Anthropic provider native Messages API pass-through."""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Self, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from anthropic import transform_schema
from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage
from anthropic.types.beta import BetaMCPToolUseBlock, BetaMessage, BetaThinkingBlock, BetaUsage
from pydantic import BaseModel

from any_llm.exceptions import InvalidRequestError, UnsupportedParameterError
from any_llm.providers.anthropic.anthropic import AnthropicProvider
from any_llm.providers.anthropic.base import BaseAnthropicProvider, _messages_betas, _pop_anthropic_beta_header
from any_llm.types.completion import CompletionParams
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
from any_llm.types.messages import (
    ThinkingBlock as AnyLLMThinkingBlock,
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


def _sdk_message_response() -> dict[str, Any]:
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "content": [{"type": "text", "text": "Hello!"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


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


def test_convert_native_message_to_response_beta_only_block() -> None:
    block = BetaMCPToolUseBlock(
        id="mcp_tool_1",
        input={"query": "test"},
        name="search",
        server_name="test-server",
        type="mcp_tool_use",
    )
    message = BetaMessage(
        id="msg_beta",
        type="message",
        role="assistant",
        model="claude-opus-5",
        stop_reason="tool_use",
        stop_sequence=None,
        content=[block],
        usage=BetaUsage(input_tokens=1, output_tokens=1),
    )

    result = BaseAnthropicProvider._convert_native_message_to_response(cast("Message", message))

    assert isinstance(result.content[0], BetaMCPToolUseBlock)
    assert result.content[0].server_name == "test-server"


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
    assert isinstance(result.content[0], AnyLLMThinkingBlock)
    assert result.content[0].type == "thinking"
    assert result.content[0].thinking == "Let me reason..."
    assert result.content[1].type == "text"
    assert result.content[1].text == "The answer is 42."


def test_convert_native_beta_message_to_response_thinking() -> None:
    message = BetaMessage(
        id="msg_beta",
        type="message",
        role="assistant",
        model="claude-opus-5",
        stop_reason="end_turn",
        stop_sequence=None,
        content=[BetaThinkingBlock(type="thinking", thinking="Let me reason...", signature="sig")],
        usage=BetaUsage(input_tokens=1, output_tokens=1),
    )

    result = BaseAnthropicProvider._convert_native_message_to_response(cast("Message", message))

    assert isinstance(result.content[0], AnyLLMThinkingBlock)
    assert result.content[0].signature == "sig"


@pytest.mark.parametrize(
    "block",
    [
        ThinkingBlock(type="thinking", thinking="Anthropic", signature="sig"),
        BetaThinkingBlock(type="thinking", thinking="Beta", signature="sig"),
    ],
)
def test_message_response_normalizes_embedded_sdk_thinking_block(block: Any) -> None:
    response = MessageResponse.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "stop_reason": "end_turn",
            "content": [block],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )

    assert isinstance(response.content[0], AnyLLMThinkingBlock)
    assert response.content[0].signature == "sig"


def test_message_response_defaults_missing_thinking_signature() -> None:
    response = MessageResponse.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "Let me reason..."}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )

    assert isinstance(response.content[0], AnyLLMThinkingBlock)
    assert response.content[0].signature == ""


def test_content_block_start_event_normalizes_sdk_thinking_block() -> None:
    event = ContentBlockStartEvent.model_validate(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": ThinkingBlock(type="thinking", thinking="Let me reason...", signature="sig"),
        }
    )

    assert isinstance(event.content_block, AnyLLMThinkingBlock)
    assert event.content_block.signature == "sig"


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
        container="container_123",
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
    assert call_kwargs["container"] == "container_123"


@pytest.mark.asyncio
async def test_anthropic_sdk_accepts_completion_sampling_parameters() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_sdk_message_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = AnthropicProvider(api_key="test-key", http_client=http_client)
        await provider._acompletion(
            CompletionParams(
                model_id="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
            )
        )

    assert len(requests) == 1
    request_body = json.loads(requests[0].content)
    assert request_body["temperature"] == 0.7
    assert request_body["top_p"] == 0.9


@pytest.mark.asyncio
async def test_anthropic_sdk_accepts_native_messages_parameters() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_sdk_message_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = AnthropicProvider(api_key="test-key", http_client=http_client)
        result = await provider._amessages(
            MessagesParams(
                model="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                container="container_123",
                service_tier="standard_only",
            )
        )

    assert isinstance(result, MessageResponse)
    assert len(requests) == 1
    request_body = json.loads(requests[0].content)
    assert request_body["temperature"] == 0.7
    assert request_body["top_p"] == 0.9
    assert request_body["top_k"] == 40
    assert request_body["container"] == "container_123"
    assert request_body["service_tier"] == "standard_only"


@pytest.mark.asyncio
async def test_amessages_rejects_prompt_cache_key_before_client_call() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", http_client=http_client)
    try:
        with pytest.raises(UnsupportedParameterError, match="prompt_cache_key"):
            await provider.amessages(
                model="claude-opus-5",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1024,
                prompt_cache_key="tenant-1",
            )
    finally:
        await http_client.aclose()

    assert requests == []


@pytest.mark.asyncio
async def test_acompletion_rejects_prompt_cache_key_before_client_call() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", http_client=http_client)
    try:
        with pytest.raises(UnsupportedParameterError, match="prompt_cache_key"):
            await provider.acompletion(
                model="claude-opus-5",
                messages=[{"role": "user", "content": "Hello"}],
                prompt_cache_key="tenant-1",
            )
    finally:
        await http_client.aclose()

    assert requests == []


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


def test_messages_betas_warns_for_unknown_edit_without_explicit_betas(
    caplog: pytest.LogCaptureFixture,
) -> None:
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        context_management={"edits": [{"type": "clear_future_thing_2027"}]},
    )

    with caplog.at_level(logging.WARNING, logger="any_llm"):
        betas = _messages_betas(params)

    assert betas == []
    assert "clear_future_thing_2027" in caplog.text
    assert "pass betas explicitly" in caplog.text


def test_messages_betas_does_not_warn_for_unknown_edit_with_explicit_betas(
    caplog: pytest.LogCaptureFixture,
) -> None:
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        context_management={"edits": [{"type": "clear_future_thing_2027"}]},
        betas=["future-beta"],
    )

    with caplog.at_level(logging.WARNING, logger="any_llm"):
        betas = _messages_betas(params)

    assert betas == ["future-beta"]
    assert caplog.text == ""


def test_messages_betas_does_not_warn_for_recognized_edit(caplog: pytest.LogCaptureFixture) -> None:
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        context_management={"edits": [{"type": "compact_20260112"}]},
    )

    with caplog.at_level(logging.WARNING, logger="any_llm"):
        betas = _messages_betas(params)

    assert betas == ["compact-2026-01-12"]
    assert caplog.text == ""


@pytest.mark.parametrize("edit", [{"value": "custom"}, {"type": 5}])
def test_messages_betas_ignores_edit_without_string_type(
    edit: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        context_management={"edits": [edit]},
    )

    with caplog.at_level(logging.WARNING, logger="any_llm"):
        betas = _messages_betas(params)

    assert betas == []
    assert caplog.text == ""


def test_messages_betas_accepts_minimum_compaction_trigger() -> None:
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        context_management={
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": 50_000},
                }
            ]
        },
    )

    assert _messages_betas(params) == ["compact-2026-01-12"]


@pytest.mark.parametrize("value", [1, 49_999, True, 50_000.0, "50000", None])
def test_messages_betas_rejects_invalid_compaction_trigger_value(value: Any) -> None:
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        context_management={
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": value},
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="trigger value must be an integer greater than or equal to 50000"):
        _messages_betas(params)


def test_messages_betas_does_not_validate_unknown_compaction_trigger_type() -> None:
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        context_management={
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {"type": "future_trigger", "value": 1},
                }
            ]
        },
    )

    assert _messages_betas(params) == ["compact-2026-01-12"]


@pytest.mark.parametrize("edits", [None, {"type": "compact_20260112"}])
def test_messages_betas_rejects_non_list_edits(edits: Any) -> None:
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        context_management={"edits": edits},
    )

    with pytest.raises(ValueError, match=r"context_management\.edits must be a list"):
        _messages_betas(params)


def test_pop_anthropic_beta_header_decodes_bytes() -> None:
    kwargs = {
        "extra_headers": {
            "anthropic-beta": b"fast-mode-2026-02-01, compact-2026-01-12",
            "x-custom-header": "custom-value",
        }
    }

    betas = _pop_anthropic_beta_header(kwargs)

    assert betas == ["fast-mode-2026-02-01", "compact-2026-01-12"]
    assert kwargs == {"extra_headers": {"x-custom-header": "custom-value"}}


@pytest.mark.parametrize("value", [object(), b"\xff"])
def test_pop_anthropic_beta_header_preserves_unparseable_values(value: object) -> None:
    kwargs = {"extra_headers": {"anthropic-beta": value}}

    betas = _pop_anthropic_beta_header(kwargs)

    assert betas == []
    assert kwargs["extra_headers"]["anthropic-beta"] is value


@pytest.mark.asyncio
async def test_amessages_merges_beta_extra_header_with_inferred_betas() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
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

    extra_headers = {
        "Anthropic-Beta": "fast-mode-2026-02-01, compact-2026-01-12",
        "x-custom-header": "custom-value",
    }
    original_extra_headers = extra_headers.copy()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", http_client=http_client)
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        context_management={"edits": [{"type": "compact_20260112"}]},
    )

    try:
        await provider._amessages(params, extra_headers=extra_headers)
    finally:
        await http_client.aclose()

    request = requests[0]
    assert request.url.query == b"beta=true"
    assert request.headers["anthropic-beta"] == "compact-2026-01-12,fast-mode-2026-02-01"
    assert request.headers["x-custom-header"] == "custom-value"
    assert extra_headers == original_extra_headers


@pytest.mark.asyncio
async def test_amessages_routes_beta_extra_header_through_beta_resource() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
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
    )

    try:
        await provider._amessages(params, extra_headers={"anthropic-beta": "fast-mode-2026-02-01"})
    finally:
        await http_client.aclose()

    assert requests[0].url.query == b"beta=true"
    assert requests[0].headers["anthropic-beta"] == "fast-mode-2026-02-01"


@pytest.mark.asyncio
async def test_amessages_beta_extra_header_suppresses_unknown_edit_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
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
        context_management={"edits": [{"type": "clear_future_thing_2027"}]},
    )

    try:
        with caplog.at_level(logging.WARNING, logger="any_llm"):
            await provider._amessages(
                params,
                extra_headers={"anthropic-beta": "future-context-management-2027-01-01"},
            )
    finally:
        await http_client.aclose()

    assert caplog.text == ""


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
                        "container": {
                            "id": "container_1",
                            "expires_at": "2026-08-05T00:00:00Z",
                            "skills": [{"skill_id": "pdf", "type": "anthropic", "version": "latest"}],
                        },
                        "usage": {"input_tokens": 10, "output_tokens": 0, "speed": "fast"},
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
    message_start = collected[0]
    assert isinstance(message_start, MessageStartEvent)
    assert message_start.message.usage.speed == "fast"
    assert message_start.message.container is not None
    assert message_start.message.container.skills is not None
    assert message_start.message.container.skills[0].skill_id == "pdf"

    content_start = collected[1]
    assert isinstance(content_start, ContentBlockStartEvent)
    assert content_start.content_block.type == "compaction"

    content_delta = collected[2]
    assert isinstance(content_delta, ContentBlockDeltaEvent)
    assert isinstance(content_delta.delta, CompactionDelta)
    assert content_delta.delta.type == "compaction_delta"
    assert content_delta.delta.content == "Conversation summary"

    content_stop = collected[3]
    assert isinstance(content_stop, ContentBlockStopEvent)
    assert content_stop.content_block is not None
    assert content_stop.content_block.type == "compaction"
    assert content_stop.content_block.content == "Conversation summary"

    message_delta = collected[4]
    assert isinstance(message_delta, MessageDeltaEvent)
    assert message_delta.delta.stop_reason == "compaction"


@pytest.mark.asyncio
async def test_amessages_streams_beta_only_content_block() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.query == b"beta=true"
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_mcp",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-opus-5",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "content": [],
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "mcp_tool_use",
                        "id": "mcp_tool_1",
                        "input": {"query": "test"},
                        "name": "search",
                        "server_name": "test-server",
                    },
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
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
        betas=["mcp-client-2025-04-04"],
    )

    try:
        stream = await provider._amessages(params)
        assert isinstance(stream, AsyncIterator)
        collected = [event async for event in stream]
    finally:
        await http_client.aclose()

    content_start = collected[1]
    assert isinstance(content_start, ContentBlockStartEvent)
    assert isinstance(content_start.content_block, BetaMCPToolUseBlock)
    assert content_start.content_block.server_name == "test-server"

    content_stop = collected[2]
    assert isinstance(content_stop, ContentBlockStopEvent)
    assert isinstance(content_stop.content_block, BetaMCPToolUseBlock)
    assert content_stop.content_block.input == {"query": "test"}


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
async def test_amessages_bare_output_format_object_is_nested_before_create() -> None:
    """The bare format object is wrapped before the native call, which requires the nesting.

    The bridge accepts either shape, so the field would otherwise mean two different things
    depending on which provider served the request.
    """
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
        output_format={"type": "json_schema", "schema": {"type": "object"}},
    )
    await BaseAnthropicProvider._amessages(provider, params)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}


@pytest.mark.asyncio
async def test_amessages_effort_only_output_config_reaches_create() -> None:
    """Every output_config field is optional, so effort alone is a valid native request."""
    mock_message = _make_message(content=[TextBlock(type="text", text="Paris")])

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
        output_format={"effort": "high"},
    )
    await BaseAnthropicProvider._amessages(provider, params)

    assert mock_client.messages.create.call_args.kwargs["output_config"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_amessages_non_object_format_raises() -> None:
    """A format value that is not an object is rejected rather than re-nested."""
    mock_client = Mock()
    mock_client.messages.create = AsyncMock()
    mock_client.messages.parse = AsyncMock()

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Capital of France?"}],
        max_tokens=1024,
        output_format={"format": "json_schema"},
    )
    with pytest.raises(InvalidRequestError, match="non-object format value"):
        await BaseAnthropicProvider._amessages(provider, params)
    mock_client.messages.create.assert_not_called()


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
async def test_amessages_output_config_dict_streams_with_anthropic_fields() -> None:
    output_config = {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    stream_result = AsyncMock()
    mock_client = Mock()
    mock_client.beta.messages.create = AsyncMock(
        return_value=_make_message(content=[TextBlock(type="text", text="{}")])
    )

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._stream_messages_async = Mock(return_value=stream_result)
    provider._convert_native_message_to_response = BaseAnthropicProvider._convert_native_message_to_response
    context_management = {"edits": [{"type": "compact_20260112"}]}
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Capital of France?"}],
        max_tokens=1024,
        stream=True,
        output_format=output_config,
        context_management=context_management,
        betas=["compact-2026-01-12"],
        cache_control={"type": "ephemeral"},
    )

    result = await BaseAnthropicProvider._amessages(provider, params)

    assert result is stream_result
    provider._stream_messages_async.assert_called_once()
    call_kwargs = provider._stream_messages_async.call_args.kwargs
    assert call_kwargs["use_beta"] is True
    assert call_kwargs["output_config"] == output_config
    assert call_kwargs["context_management"] == context_management
    assert call_kwargs["betas"] == ["compact-2026-01-12"]
    assert call_kwargs["cache_control"] == {"type": "ephemeral"}
    mock_client.beta.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_amessages_bare_output_config_is_normalized_for_streaming() -> None:
    output_format = {"type": "json_schema", "schema": {"type": "object"}}
    stream_result = AsyncMock()
    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = Mock()
    provider._stream_messages_async = Mock(return_value=stream_result)
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Capital of France?"}],
        max_tokens=1024,
        stream=True,
        output_format=output_format,
    )

    result = await BaseAnthropicProvider._amessages(provider, params)

    assert result is stream_result
    call_kwargs = provider._stream_messages_async.call_args.kwargs
    assert call_kwargs["output_config"] == {"format": output_format}


@pytest.mark.asyncio
async def test_amessages_typed_output_format_streams_with_sdk_parser() -> None:
    class City(BaseModel):
        city: str

    stream_result = AsyncMock()
    mock_client = Mock()
    mock_client.messages.parse = AsyncMock(
        return_value=_make_message(content=[TextBlock(type="text", text='{"city": "Paris"}')])
    )

    provider = Mock(spec=BaseAnthropicProvider)
    provider.client = mock_client
    provider._stream_messages_async = Mock(return_value=stream_result)
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Capital of France?"}],
        max_tokens=1024,
        stream=True,
        output_format=City,
    )

    result = await BaseAnthropicProvider._amessages(provider, params)

    assert result is stream_result
    call_kwargs = provider._stream_messages_async.call_args.kwargs
    assert call_kwargs["use_beta"] is False
    assert call_kwargs["output_format"] is City
    mock_client.messages.parse.assert_not_called()


@pytest.mark.asyncio
async def test_anthropic_provider_allows_streaming_output_format() -> None:
    class City(BaseModel):
        city: str

    async def events() -> AsyncIterator[MessageStopEvent]:
        yield MessageStopEvent(type="message_stop")

    with patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")
    provider._stream_messages_async = Mock(return_value=events())  # type: ignore[method-assign]

    result = await provider.amessages(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Capital of France?"}],
        max_tokens=1024,
        stream=True,
        output_format=City,
    )
    collected = [event async for event in cast("AsyncIterator[MessageStopEvent]", result)]

    assert [event.type for event in collected] == ["message_stop"]


@pytest.mark.asyncio
async def test_amessages_typed_output_format_streams_through_sdk_transport() -> None:
    class City(BaseModel):
        city: str

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_structured",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-3-5-sonnet",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "content": [],
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": '{"city":"Paris"}'},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        body = "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", http_client=http_client)
    try:
        result = await provider.amessages(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Capital of France?"}],
            max_tokens=1024,
            stream=True,
            output_format=City,
        )
        collected = [event async for event in cast("AsyncIterator[Any]", result)]
    finally:
        await http_client.aclose()

    assert collected[-1].type == "message_stop"
    assert len(requests) == 1
    request_body = json.loads(requests[0].content)
    assert request_body["output_config"] == {
        "format": {"type": "json_schema", "schema": transform_schema(City.model_json_schema())}
    }


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
async def test_amessages_stream_preserves_accumulated_stop_event_payloads() -> None:
    """The SDK's stream helper attaches the accumulated message and block to the stop events."""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.query == b""
        request_body = json.loads(request.content)
        assert request_body["temperature"] == 0.7
        assert request_body["top_p"] == 0.9
        assert request_body["top_k"] == 40
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_accumulated",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-opus-5",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "content": [],
                        "usage": {"input_tokens": 4, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello!"}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 2},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        body = "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", http_client=http_client)
    params = MessagesParams(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        stream=True,
    )

    collected: list[Any] = []
    try:
        stream = await provider._amessages(params)
        assert isinstance(stream, AsyncIterator)
        async for event in stream:
            collected.append(event)
    finally:
        await http_client.aclose()

    block_stop = next(e for e in collected if isinstance(e, ContentBlockStopEvent))
    assert block_stop.content_block is not None
    assert block_stop.content_block.type == "text"

    message_stop = next(e for e in collected if isinstance(e, MessageStopEvent))
    assert message_stop.message is not None
    assert message_stop.message.id == "msg_accumulated"
    assert message_stop.message.stop_reason == "end_turn"
    assert message_stop.message.usage.output_tokens == 2


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


@pytest.mark.asyncio
async def test_amessages_without_timeout_translates_nonstreaming_guard() -> None:
    """The messages API surfaces the same actionable error as completions for the pre-flight guard.

    Uses a real ``AsyncAnthropic`` client so the SDK guard runs; the transport is patched to fail
    fast so the test stays hermetic if the guard ever changes.
    """
    provider = AnthropicProvider(api_key="sk-test")

    with (
        patch.object(
            provider.client, "post", new=AsyncMock(side_effect=AssertionError("network should not be reached"))
        ),
        pytest.raises(InvalidRequestError) as exc_info,
    ):
        await provider.amessages(
            model="claude-opus-5",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=65536,
            stream=False,
        )

    message = str(exc_info.value)
    assert "timeout" in message
    assert "stream=True" in message
