"""Tests for messages()/amessages() SDK API."""

import pytest

from any_llm.api import amessages


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
