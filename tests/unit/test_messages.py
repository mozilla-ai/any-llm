"""Tests for messages()/amessages() SDK API."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.api import amessages
from any_llm.types.messages import MessageResponse


@pytest.mark.asyncio
async def test_messages_invalid_model_format_no_separator() -> None:
    """Test amessages raises ValueError for model without separator."""
    with pytest.raises(
        ValueError,
        match=r"Invalid model format. Expected 'provider:model' or 'provider/model', got 'claude-3'",
    ):
        await amessages(
            "claude-3",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )


@pytest.mark.asyncio
async def test_messages_invalid_model_format_empty_provider() -> None:
    """Test amessages raises ValueError for model with empty provider."""
    with pytest.raises(ValueError, match="Invalid model format"):
        await amessages(
            ":claude-3",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )


@pytest.mark.asyncio
async def test_messages_invalid_model_format_empty_model() -> None:
    """Test amessages raises ValueError for model with empty model name."""
    with pytest.raises(ValueError, match="Invalid model format"):
        await amessages(
            "anthropic:",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )


@pytest.mark.asyncio
async def test_messages_unsupported_provider() -> None:
    """Test amessages raises error for an unsupported provider."""
    with pytest.raises((ValueError, Exception)):
        await amessages(
            "nonexistent:model-name",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )


def test_messages_params_model_creation() -> None:
    """Test MessagesParams creates correctly with required fields."""
    from any_llm.types.messages import MessagesParams

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
    )
    assert params.model == "claude-3-5-sonnet"
    assert params.max_tokens == 1024
    assert len(params.messages) == 1
    assert params.stream is None
    assert params.system is None


def test_messages_params_with_all_fields() -> None:
    """Test MessagesParams with all optional fields populated."""
    from any_llm.types.messages import MessagesParams

    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=4096,
        system="You are helpful.",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        stream=True,
        stop_sequences=["END"],
        tools=[{"name": "test", "description": "test", "input_schema": {"type": "object", "properties": {}}}],
        tool_choice={"type": "auto"},
        metadata={"user_id": "user-1"},
        thinking={"type": "enabled", "budget_tokens": 8192},
    )
    assert params.system == "You are helpful."
    assert params.temperature == 0.7
    assert params.stream is True
    assert params.stop_sequences == ["END"]
    assert params.tools is not None
    assert len(params.tools) == 1


def test_messages_params_rejects_extra_fields() -> None:
    """Test MessagesParams rejects unknown fields (extra='forbid')."""
    from pydantic import ValidationError

    from any_llm.types.messages import MessagesParams

    with pytest.raises(ValidationError):
        MessagesParams(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
            unknown_field="value",  # type: ignore[call-arg]
        )


def test_message_response_model() -> None:
    """Test MessageResponse creation."""
    from any_llm.types.messages import MessageContentBlock, MessageResponse, MessageUsage

    resp = MessageResponse(
        id="msg_123",
        type="message",
        role="assistant",
        content=[MessageContentBlock(type="text", text="Hello!")],
        model="claude-3-5-sonnet",
        stop_reason="end_turn",
        usage=MessageUsage(input_tokens=10, output_tokens=5),
    )
    assert resp.id == "msg_123"
    assert resp.content[0].text == "Hello!"
    assert resp.usage.input_tokens == 10


def test_message_stream_event_model() -> None:
    """Test MessageStreamEvent creation."""
    from any_llm.types.messages import MessageStreamEvent

    event = MessageStreamEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "text_delta", "text": "Hello"},
    )
    assert event.type == "content_block_delta"
    assert event.index == 0
    assert event.delta is not None
    assert event.delta["text"] == "Hello"


def test_messages_exported_from_init() -> None:
    """Test that messages and amessages are exported from any_llm."""
    import any_llm

    assert hasattr(any_llm, "messages")
    assert hasattr(any_llm, "amessages")
    assert "messages" in any_llm.__all__
    assert "amessages" in any_llm.__all__


@pytest.mark.asyncio
async def test_amessages_parameter_capture() -> None:
    """Test that amessages correctly captures and passes all parameters."""
    mock_provider = Mock()
    mock_provider.amessages = AsyncMock(return_value=Mock())

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        await amessages(
            model="openai:gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
            system="Be helpful",
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop_sequences=["END"],
            tools=[{"name": "test", "input_schema": {}}],
            tool_choice={"type": "auto"},
            metadata={"user_id": "u1"},
            thinking={"type": "enabled", "budget_tokens": 4096},
            api_key="sk-test",
            api_base="https://custom.example.com",
        )

        from any_llm.constants import LLMProvider

        mock_create.assert_called_once_with(
            LLMProvider.OPENAI,
            api_key="sk-test",
            api_base="https://custom.example.com",
        )

        mock_provider.amessages.assert_called_once()
        call_args = mock_provider.amessages.call_args
        assert call_args.kwargs["model"] == "gpt-4"
        assert call_args.kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_args.kwargs["max_tokens"] == 1024
        assert call_args.kwargs["system"] == "Be helpful"
        assert call_args.kwargs["temperature"] == 0.7
        assert call_args.kwargs["top_p"] == 0.9
        assert call_args.kwargs["top_k"] == 40
        assert call_args.kwargs["stop_sequences"] == ["END"]
        assert call_args.kwargs["tools"] == [{"name": "test", "input_schema": {}}]
        assert call_args.kwargs["tool_choice"] == {"type": "auto"}
        assert call_args.kwargs["metadata"] == {"user_id": "u1"}
        assert call_args.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}


@pytest.mark.asyncio
async def test_amessages_with_explicit_provider() -> None:
    """Test amessages with explicit provider parameter."""
    mock_provider = Mock()
    mock_provider.amessages = AsyncMock(return_value=Mock())

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        await amessages(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            provider="openai",
        )

        from any_llm.constants import LLMProvider

        mock_create.assert_called_once_with(LLMProvider.OPENAI, api_key=None, api_base=None)
        call_args = mock_provider.amessages.call_args
        assert call_args.kwargs["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_amessages_with_client_args() -> None:
    """Test amessages passes client_args to provider creation."""
    mock_provider = Mock()
    mock_provider.amessages = AsyncMock(return_value=Mock())

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        await amessages(
            model="openai:gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            client_args={"timeout": 30},
        )

        from any_llm.constants import LLMProvider

        mock_create.assert_called_once_with(LLMProvider.OPENAI, api_key=None, api_base=None, timeout=30)


@pytest.mark.asyncio
async def test_default_amessages_non_streaming() -> None:
    """Test default _amessages implementation converts Completions to Messages format."""
    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionMessage,
        Choice,
        CompletionUsage,
    )
    from any_llm.types.messages import MessageResponse, MessagesParams

    mock_completion = ChatCompletion(
        id="chatcmpl-test",
        model="gpt-4",
        created=0,
        object="chat.completion",
        choices=[Choice(index=0, finish_reason="stop", message=ChatCompletionMessage(role="assistant", content="Hi!"))],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    mock_provider = Mock()
    mock_provider._acompletion = AsyncMock(return_value=mock_completion)

    from any_llm.any_llm import AnyLLM

    params = MessagesParams(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )
    result = await AnyLLM._amessages(mock_provider, params)
    assert isinstance(result, MessageResponse)
    assert result.id == "chatcmpl-test"
    assert result.content[0].type == "text"
    assert result.content[0].text == "Hi!"
    assert result.stop_reason == "end_turn"
    assert result.usage.input_tokens == 10


@pytest.mark.asyncio
async def test_default_amessages_streaming() -> None:
    """Test default _amessages streaming converts ChatCompletionChunks to MessageStreamEvents."""
    from any_llm.types.completion import (
        ChatCompletionChunk,
        ChoiceDelta,
        ChunkChoice,
    )
    from any_llm.types.messages import MessagesParams

    async def mock_stream() -> Any:
        yield ChatCompletionChunk(
            id="chunk-1",
            model="gpt-4",
            created=0,
            object="chat.completion.chunk",
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="Hello"), finish_reason=None)],
        )
        yield ChatCompletionChunk(
            id="chunk-2",
            model="gpt-4",
            created=0,
            object="chat.completion.chunk",
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
        )

    mock_provider = Mock()
    mock_provider._acompletion = AsyncMock(return_value=mock_stream())

    from any_llm.any_llm import AnyLLM

    params = MessagesParams(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        stream=True,
    )
    result = await AnyLLM._amessages(mock_provider, params)
    assert not isinstance(result, MessageResponse)

    events = []
    async for event in result:
        events.append(event)

    types = [e.type for e in events]
    assert "message_start" in types
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert "message_delta" in types
    assert "message_stop" in types


def test_supports_messages_flag() -> None:
    """Test that SUPPORTS_MESSAGES defaults to True."""
    from any_llm.any_llm import AnyLLM

    assert AnyLLM.SUPPORTS_MESSAGES is True


def test_sync_messages_returns_message_response() -> None:
    """Test that the sync messages() wrapper returns MessageResponse for non-streaming."""
    from any_llm.types.messages import MessageContentBlock, MessageUsage

    mock_response = MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        content=[MessageContentBlock(type="text", text="Hi!")],
        model="test-model",
        stop_reason="end_turn",
        usage=MessageUsage(input_tokens=5, output_tokens=3),
    )

    mock_provider = Mock()
    mock_provider.messages = Mock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        from any_llm.api import messages as sync_messages

        result = sync_messages(
            model="openai:gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )

    assert isinstance(result, MessageResponse)
    assert result.content[0].text == "Hi!"


@pytest.mark.asyncio
async def test_amessages_constructs_params_correctly() -> None:
    """Test that AnyLLM.amessages builds MessagesParams and delegates to _amessages."""
    from any_llm.any_llm import AnyLLM
    from any_llm.types.messages import MessagesParams

    mock_response = Mock(spec=MessageResponse)
    mock_provider = Mock()
    mock_provider._amessages = AsyncMock(return_value=mock_response)

    result = await AnyLLM.amessages(
        mock_provider,
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=512,
        system="Be helpful",
        temperature=0.5,
    )
    assert result is mock_response

    call_args = mock_provider._amessages.call_args
    params = call_args.args[0]
    assert isinstance(params, MessagesParams)
    assert params.model == "test-model"
    assert params.max_tokens == 512
    assert params.system == "Be helpful"
    assert params.temperature == 0.5
