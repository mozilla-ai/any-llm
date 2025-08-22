from typing import Any

import pytest
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig
from any_llm.providers.cohere.utils import _patch_messages
from any_llm.types.completion import CompletionParams


def _mk_provider() -> Any:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    return CohereProvider(ApiConfig(api_key="test-api-key"))


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
        await provider.acompletion(
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
        await provider.acompletion(
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
    with pytest.raises(ValueError, match="A tool message must be preceded by an assistant message with tool_calls."):
        _patch_messages(messages)
