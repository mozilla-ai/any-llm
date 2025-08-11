from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig
from any_llm.providers.anthropic.anthropic import AnthropicProvider


@contextmanager
def mock_anthropic_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.anthropic.anthropic.Anthropic") as mock_anthropic,
        patch("any_llm.providers.anthropic.anthropic._convert_response"),
    ):
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = Mock()
        yield mock_anthropic


def test_anthropic_client_created_with_api_key_and_api_base() -> None:
    """Test that Anthropic client is created with api_key and api_base as base_url when provided."""
    api_key = "test-api-key"
    custom_endpoint = "https://custom-anthropic-endpoint"

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))
        provider.completion("model-id", [{"role": "user", "content": "Hello"}])

        mock_anthropic.assert_called_once_with(api_key=api_key, base_url=custom_endpoint)


def test_anthropic_client_created_without_api_base() -> None:
    """Test that Anthropic client is created with None base_url when api_base is not provided."""
    api_key = "test-api-key"

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        provider.completion("model-id", [{"role": "user", "content": "Hello"}])

        mock_anthropic.assert_called_once_with(api_key=api_key, base_url=None)


def testcompletion_with_system_message() -> None:
    """Test that completion correctly processes a system message."""
    api_key = "test-api-key"
    model = "model-id"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        provider.completion(model, messages)

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant.",
            max_tokens=4096,
        )


def testcompletion_with_multiple_system_messages() -> None:
    """Test that completion concatenates multiple system messages."""
    api_key = "test-api-key"
    model = "model-id"
    messages = [
        {"role": "system", "content": "First part."},
        {"role": "system", "content": "Second part."},
        {"role": "user", "content": "Hello"},
    ]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        provider.completion(model, messages)

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            system="First part.\nSecond part.",
            max_tokens=4096,
        )


def testcompletion_with_kwargs() -> None:
    """Test that completion passes kwargs to the Anthropic client."""
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]
    kwargs = {"temperature": 0.5, "max_tokens": 100}

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        provider.completion(model, messages, **kwargs)

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=messages,
            **kwargs,
        )


def testcompletion_with_tool_choice_required() -> None:
    """Test that completion correctly processes tool_choice='required'."""
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]
    kwargs = {"tool_choice": "required"}

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        provider.completion(model, messages, **kwargs)

        expected_kwargs = {"tool_choice": {"type": "any", "disable_parallel_tool_use": False}}

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=messages,
            max_tokens=4096,
            **expected_kwargs,
        )


@pytest.mark.parametrize("parallel_tool_calls", [True, False])
def testcompletion_with_tool_choice_and_parallel_tool_calls(parallel_tool_calls: bool) -> None:
    """Test that completion correctly processes tool_choice and parallel_tool_calls."""
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]
    kwargs = {"tool_choice": "auto", "parallel_tool_calls": parallel_tool_calls}

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        provider.completion(model, messages, **kwargs)

        expected_kwargs = {"tool_choice": {"type": "auto", "disable_parallel_tool_use": not parallel_tool_calls}}

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=4096,
            **expected_kwargs,
        )


def test_stream_with_response_format_raises() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    provider = AnthropicProvider(ApiConfig(api_key=api_key))

    with pytest.raises(UnsupportedParameterError):
        next(
            provider._stream_completion(
                client=Mock(),
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        )
