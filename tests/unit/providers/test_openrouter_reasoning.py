"""Unit tests for OpenRouter reasoning support."""

from unittest.mock import AsyncMock, patch

import pytest

from any_llm.provider import ApiConfig, ProviderFactory
from any_llm.providers.openrouter import OpenrouterProvider
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionParams,
    Reasoning,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up test environment variables."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-123")


def test_provider_reasoning_support() -> None:
    """Test that OpenRouter provider has reasoning support enabled."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))
    assert provider.SUPPORTS_COMPLETION_REASONING is True


@pytest.mark.asyncio
async def test_reasoning_param_added_with_explicit_effort() -> None:
    """Test that reasoning parameter is added when reasoning_effort is specified."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

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

    with patch(
        "any_llm.providers.openrouter.openrouter.AsyncOpenAI"
    ) as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="low",
        )

        await provider.acompletion(params)

        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "low"


@pytest.mark.asyncio
async def test_reasoning_param_with_include_reasoning_flag() -> None:
    """Test that reasoning is included when include_reasoning=True is passed."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

    mock_completion = ChatCompletion(
        id="test-789",
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
                    reasoning=Reasoning(content="Thinking..."),
                ),
            )
        ],
    )

    with patch(
        "any_llm.providers.openrouter.openrouter.AsyncOpenAI"
    ) as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort=None,
        )

        await provider.acompletion(params, include_reasoning=True)

        call_args = mock_client.chat.completions.create.call_args
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["reasoning"] == {}


@pytest.mark.asyncio
async def test_no_reasoning_param_by_default() -> None:
    """Test that reasoning is not included by default."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

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

    with patch(
        "any_llm.providers.openrouter.openrouter.AsyncOpenAI"
    ) as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort=None,
        )

        await provider.acompletion(params)

        call_args = mock_client.chat.completions.create.call_args
        if "extra_body" in call_args.kwargs:
            assert not call_args.kwargs["extra_body"]
        else:
            assert "extra_body" not in call_args.kwargs


def test_factory_integration() -> None:
    """Test that OpenRouter can be created via factory and has reasoning support."""
    provider = ProviderFactory.create_provider(
        "openrouter", ApiConfig(api_key="sk-test")
    )
    assert isinstance(provider, OpenrouterProvider)
    assert provider.SUPPORTS_COMPLETION_REASONING is True
