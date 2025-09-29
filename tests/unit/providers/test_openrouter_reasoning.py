"""Unit tests for OpenRouter reasoning support."""

from unittest.mock import AsyncMock, patch

import pytest

from any_llm.providers.openrouter import OpenrouterProvider
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionParams,
    Reasoning,
)


@pytest.mark.asyncio
async def test_reasoning_param_added_with_explicit_effort() -> None:
    """Test that reasoning parameter is added when reasoning_effort is specified."""
    mock_completion = ChatCompletion(
        id="test-123",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Hello!",
                    reasoning=Reasoning(content="Let me think..."),
                ),
            )
        ],
    )

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="low",
        )

        provider = OpenrouterProvider(api_key="sk-test")
        await provider._acompletion(params)

        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "low"


@pytest.mark.asyncio
async def test_reasoning_auto_excludes_reasoning() -> None:
    """Test that reasoning_effort='auto' does not include reasoning - no extra_body at all."""
    mock_completion = ChatCompletion(
        id="test-456",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="Hello!"),
            )
        ],
    )

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        provider = OpenrouterProvider(api_key="sk-test")
        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        await provider._acompletion(params)

        call_args = mock_client.chat.completions.create.call_args
        # Should not have extra_body at all when "auto"
        assert "extra_body" not in call_args.kwargs


@pytest.mark.asyncio
async def test_reasoning_with_custom_reasoning_object() -> None:
    """Test that custom reasoning object overrides reasoning_effort."""
    mock_completion = ChatCompletion(
        id="test-custom",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Hello!",
                    reasoning=Reasoning(content="Custom reasoning"),
                ),
            )
        ],
    )

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="low",  # This should be overridden
        )

        provider = OpenrouterProvider(api_key="sk-test")
        await provider._acompletion(params, reasoning={"effort": "high", "max_tokens": 1000})

        call_args = mock_client.chat.completions.create.call_args
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "high"
        assert call_args.kwargs["extra_body"]["reasoning"]["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_no_extra_body_when_no_reasoning() -> None:
    """Test that no extra_body is sent when reasoning is not requested."""
    mock_completion = ChatCompletion(
        id="test-no-extra",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="Hello!"),
            )
        ],
    )

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort=None,  # No reasoning
        )
        provider = OpenrouterProvider(api_key="sk-test")
        await provider._acompletion(params)

        call_args = mock_client.chat.completions.create.call_args
        # Should not have extra_body at all when no reasoning
        assert "extra_body" not in call_args.kwargs


@pytest.mark.asyncio
async def test_preserve_existing_extra_body() -> None:
    """Test that existing extra_body is preserved when no reasoning is added."""
    mock_completion = ChatCompletion(
        id="test-preserve",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="Hello!"),
            )
        ],
    )

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        provider = OpenrouterProvider(api_key="sk-test")
        await provider._acompletion(params, extra_body={"some_other_param": "value"})

        call_args = mock_client.chat.completions.create.call_args
        # Should preserve existing extra_body but not add reasoning
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["some_other_param"] == "value"
        assert "reasoning" not in call_args.kwargs["extra_body"]


@pytest.mark.asyncio
async def test_streaming_with_reasoning() -> None:
    """Test that streaming passes through reasoning parameters correctly."""
    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="high",
            stream=True,
        )
        provider = OpenrouterProvider(api_key="sk-test")
        await provider._acompletion(params)

        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["stream"] is True
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "high"
