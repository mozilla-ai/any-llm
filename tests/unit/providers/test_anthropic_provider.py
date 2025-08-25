from contextlib import contextmanager
from typing import Any, Literal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig
from any_llm.providers.anthropic.anthropic import AnthropicProvider
from any_llm.providers.anthropic.utils import DEFAULT_MAX_TOKENS, REASONING_EFFORT_TO_THINKING_BUDGETS
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_anthropic_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic") as mock_anthropic,
        patch("any_llm.providers.anthropic.anthropic._convert_response"),
    ):
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create = AsyncMock()
        yield mock_anthropic


@pytest.mark.asyncio
async def test_anthropic_client_created_with_api_key_and_api_base() -> None:
    """Test that Anthropic client is created with api_key and api_base as base_url when provided."""
    api_key = "test-api-key"
    custom_endpoint = "https://custom-anthropic-endpoint"

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))
        await provider.acompletion(
            CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_anthropic.assert_called_once_with(api_key=api_key, base_url=custom_endpoint)


@pytest.mark.asyncio
async def test_anthropic_client_created_without_api_base() -> None:
    """Test that Anthropic client is created with None base_url when api_base is not provided."""
    api_key = "test-api-key"

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(
            CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_anthropic.assert_called_once_with(api_key=api_key, base_url=None)


@pytest.mark.asyncio
async def test_completion_with_system_message() -> None:
    """Test that completion correctly processes a system message."""
    api_key = "test-api-key"
    model = "model-id"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=messages))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant.",
            max_tokens=DEFAULT_MAX_TOKENS,
        )


@pytest.mark.asyncio
async def test_completion_with_multiple_system_messages() -> None:
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
        await provider.acompletion(CompletionParams(model_id=model, messages=messages))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            system="First part.\nSecond part.",
            max_tokens=DEFAULT_MAX_TOKENS,
        )


@pytest.mark.asyncio
async def test_completion_with_kwargs() -> None:
    """Test that completion passes kwargs to the Anthropic client."""
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=messages, max_tokens=100, temperature=0.5))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model, messages=messages, max_tokens=100, temperature=0.5
        )


@pytest.mark.asyncio
async def test_completion_with_tool_choice_required() -> None:
    """Test that completion correctly processes tool_choice='required'."""
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=messages, tool_choice="required"))

        expected_kwargs = {"tool_choice": {"type": "any", "disable_parallel_tool_use": False}}

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            **expected_kwargs,
        )


@pytest.mark.parametrize("parallel_tool_calls", [True, False])
@pytest.mark.asyncio
async def test_completion_with_tool_choice_and_parallel_tool_calls(parallel_tool_calls: bool) -> None:
    """Test that completion correctly processes tool_choice and parallel_tool_calls."""
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(
            CompletionParams(
                model_id=model, messages=messages, tool_choice="auto", parallel_tool_calls=parallel_tool_calls
            ),
        )

        expected_kwargs = {"tool_choice": {"type": "auto", "disable_parallel_tool_use": not parallel_tool_calls}}

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            **expected_kwargs,
            max_tokens=DEFAULT_MAX_TOKENS,
        )


@pytest.mark.asyncio
async def test_completion_inside_agent_loop(agent_loop_messages: list[dict[str, Any]]) -> None:
    api_key = "test-api-key"
    model = "model-id"

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=agent_loop_messages))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[
                {"role": "user", "content": "What is the weather like in Salvaterra?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "foo", "name": "get_weather", "input": {"location": "Salvaterra"}}
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "foo", "content": "sunny"}]},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
        )


@pytest.mark.parametrize(
    "reasoning_effort",
    [
        None,
        "low",
        "medium",
        "high",
    ],
)
@pytest.mark.asyncio
async def test_completion_with_custom_reasoning_effort(
    reasoning_effort: Literal["low", "medium", "high"] | None,
) -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(
            CompletionParams(model_id=model, messages=messages, reasoning_effort=reasoning_effort)
        )

        if reasoning_effort is not None:
            expected_thinking = {
                "type": "enabled",
                "budget_tokens": REASONING_EFFORT_TO_THINKING_BUDGETS[reasoning_effort],
            }
        else:
            expected_thinking = {"type": "disabled"}

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model, messages=messages, max_tokens=DEFAULT_MAX_TOKENS, thinking=expected_thinking
        )


@pytest.mark.asyncio
async def test_response_format_raises_error() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    provider = AnthropicProvider(ApiConfig(api_key=api_key))

    with pytest.raises(UnsupportedParameterError, match="Check the following links:"):
        await provider.acompletion(
            CompletionParams(
                model_id=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        )
