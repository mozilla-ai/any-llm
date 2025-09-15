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

    with patch("any_llm.providers.openrouter.openrouter.AsyncOpenAI") as mock_client_class:
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
async def test_reasoning_auto_excludes_reasoning() -> None:
    """Test that reasoning_effort='auto' does not include reasoning - no extra_body at all."""
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

    with patch("any_llm.providers.openrouter.openrouter.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="auto",  # "auto" should exclude reasoning completely
        )

        await provider.acompletion(params)

        call_args = mock_client.chat.completions.create.call_args
        # Should not have extra_body at all when "auto"
        assert "extra_body" not in call_args.kwargs


@pytest.mark.asyncio
async def test_reasoning_with_custom_reasoning_object() -> None:
    """Test that custom reasoning object overrides reasoning_effort."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

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

    with patch("any_llm.providers.openrouter.openrouter.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="low",  # This should be overridden
        )

        # Pass custom reasoning object
        await provider.acompletion(params, reasoning={"effort": "high", "max_tokens": 1000})

        call_args = mock_client.chat.completions.create.call_args
        assert "extra_body" in call_args.kwargs
        # Custom reasoning should override reasoning_effort
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "high"
        assert call_args.kwargs["extra_body"]["reasoning"]["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_no_extra_body_when_no_reasoning() -> None:
    """Test that no extra_body is sent when reasoning is not requested."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

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

    with patch("any_llm.providers.openrouter.openrouter.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort=None,  # No reasoning
        )

        await provider.acompletion(params)

        call_args = mock_client.chat.completions.create.call_args
        # Should not have extra_body at all when no reasoning
        assert "extra_body" not in call_args.kwargs


@pytest.mark.asyncio
async def test_preserve_existing_extra_body() -> None:
    """Test that existing extra_body is preserved when no reasoning is added."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

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

    with patch("any_llm.providers.openrouter.openrouter.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="auto",  # No reasoning added
        )

        # Pass other extra_body params through kwargs
        await provider.acompletion(params, extra_body={"some_other_param": "value"})

        call_args = mock_client.chat.completions.create.call_args
        # Should preserve existing extra_body but not add reasoning
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["some_other_param"] == "value"
        assert "reasoning" not in call_args.kwargs["extra_body"]


@pytest.mark.asyncio
async def test_reasoning_with_high_effort() -> None:
    """Test that reasoning_effort='high' works correctly."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

    mock_completion = ChatCompletion(
        id="test-high",
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
                    reasoning=Reasoning(content="High effort reasoning"),
                ),
            )
        ],
    )

    with patch("any_llm.providers.openrouter.openrouter.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="high",
        )

        await provider.acompletion(params)

        call_args = mock_client.chat.completions.create.call_args
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "high"


@pytest.mark.asyncio
async def test_reasoning_with_medium_effort() -> None:
    """Test that reasoning_effort='medium' works correctly."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

    mock_completion = ChatCompletion(
        id="test-medium",
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
                    reasoning=Reasoning(content="Medium effort reasoning"),
                ),
            )
        ],
    )

    with patch("any_llm.providers.openrouter.openrouter.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="medium",
        )

        await provider.acompletion(params)

        call_args = mock_client.chat.completions.create.call_args
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "medium"


@pytest.mark.asyncio
async def test_null_reasoning_effort_ignored() -> None:
    """Test that None reasoning_effort is handled correctly."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

    mock_completion = ChatCompletion(
        id="test-null",
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

    with patch("any_llm.providers.openrouter.openrouter.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort=None,
        )

        await provider.acompletion(params)

        call_args = mock_client.chat.completions.create.call_args
        # None values should not add reasoning
        assert "extra_body" not in call_args.kwargs


@pytest.mark.asyncio
async def test_preserve_existing_extra_body_with_reasoning() -> None:
    """Test that existing extra_body is preserved when reasoning is added."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

    mock_completion = ChatCompletion(
        id="test-preserve-with-reasoning",
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
                    reasoning=Reasoning(content="Reasoning content"),
                ),
            )
        ],
    )

    with patch("any_llm.providers.openrouter.openrouter.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="medium",
        )

        # Pass other extra_body params through kwargs
        await provider.acompletion(params, extra_body={"custom_param": "test_value"})

        call_args = mock_client.chat.completions.create.call_args
        # Should preserve existing extra_body AND add reasoning
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["custom_param"] == "test_value"
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "medium"


@pytest.mark.asyncio
async def test_unknown_reasoning_effort_fallback() -> None:
    """Test that unknown reasoning_effort values do not add reasoning."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

    mock_completion = ChatCompletion(
        id="test-unknown",
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

    with patch("any_llm.providers.openrouter.openrouter.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="unknown_value",
        )

        await provider.acompletion(params)

        call_args = mock_client.chat.completions.create.call_args
        # Unknown values should not add reasoning
        assert "extra_body" not in call_args.kwargs


@pytest.mark.asyncio
async def test_custom_reasoning_overrides_effort() -> None:
    """Test that explicit kwargs['reasoning'] overrides reasoning_effort parameter."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

    mock_completion = ChatCompletion(
        id="test-override",
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
                    reasoning=Reasoning(content="Override reasoning"),
                ),
            )
        ],
    )

    with patch("any_llm.providers.openrouter.openrouter.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="low",  # This should be completely ignored
        )

        # Custom reasoning should take precedence
        custom_reasoning = {"effort": "high", "max_tokens": 2000, "enabled": True}
        await provider.acompletion(params, reasoning=custom_reasoning)

        call_args = mock_client.chat.completions.create.call_args
        assert "extra_body" in call_args.kwargs
        # Should use custom reasoning, not the effort from params
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "high"
        assert call_args.kwargs["extra_body"]["reasoning"]["max_tokens"] == 2000
        assert call_args.kwargs["extra_body"]["reasoning"]["enabled"] is True


@pytest.mark.asyncio
async def test_streaming_with_reasoning() -> None:
    """Test that streaming passes through reasoning parameters correctly."""
    provider = OpenrouterProvider(ApiConfig(api_key="sk-test"))

    with patch("any_llm.providers.openrouter.openrouter.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="high",
            stream=True,
        )

        await provider.acompletion(params)

        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["stream"] is True
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "high"


def test_factory_integration() -> None:
    """Test that OpenRouter can be created via factory and has reasoning support."""
    provider = ProviderFactory.create_provider("openrouter", ApiConfig(api_key="sk-test"))
    assert isinstance(provider, OpenrouterProvider)
    assert provider.SUPPORTS_COMPLETION_REASONING is True
