from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.messages import MessageResponse, MessageStreamEvent, ToolUseBlock
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_messages_non_streaming(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that all providers support the Messages API (non-streaming)."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")
        model_id = provider_model_map[provider]
        result = await llm.amessages(
            model=model_id,
            messages=[{"role": "user", "content": "Say hello in exactly one word."}],
            max_tokens=64,
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, MessageResponse)
    assert result.role == "assistant"
    assert len(result.content) >= 1


@pytest.mark.asyncio
async def test_messages_streaming(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that all providers support the Messages API (streaming)."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")
        if not llm.SUPPORTS_COMPLETION_STREAMING:
            pytest.skip(f"{provider.value} does not support streaming")
        model_id = provider_model_map[provider]
        result = await llm.amessages(
            model=model_id,
            messages=[{"role": "user", "content": "Say hello in exactly one word."}],
            max_tokens=64,
            stream=True,
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise

    assert isinstance(result, AsyncIterator)

    event_types: list[str] = []
    async for event in result:
        assert isinstance(event, MessageStreamEvent)
        event_types.append(event.type)

    assert "message_start" in event_types
    assert "message_stop" in event_types


def test_messages_streaming_sync(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that the sync Messages streaming path works for supported providers.

    Mirrors test_messages_streaming, but drives the sync `messages()` wrapper
    instead of `amessages()`. This is the path that regressed in #1253 (the
    response was opened and consumed on two different event loops); see #1260.
    """
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")
        if not llm.SUPPORTS_COMPLETION_STREAMING:
            pytest.skip(f"{provider.value} does not support streaming")
        model_id = provider_model_map[provider]
        result = llm.messages(
            model=model_id,
            messages=[{"role": "user", "content": "Say hello in exactly one word."}],
            max_tokens=64,
            stream=True,
        )
        assert isinstance(result, Iterator)

        # The sync bridge opens the connection lazily: request/auth/connection
        # errors surface from the first next() here, not from the messages() call
        # above, so the skip guards must cover this loop too.
        event_types: list[str] = []
        for event in result:
            assert isinstance(event, MessageStreamEvent)
            event_types.append(event.type)
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise

    assert "message_start" in event_types
    assert "message_stop" in event_types


@pytest.mark.asyncio
async def test_messages_with_system_prompt(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test Messages API with a system prompt."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")
        model_id = provider_model_map[provider]
        result = await llm.amessages(
            model=model_id,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=64,
            system="You are a math tutor. Always answer with just the number.",
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, MessageResponse)
    assert len(result.content) >= 1
    assert result.usage.input_tokens >= 0
    assert result.usage.output_tokens >= 0


_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "The city name."}},
        "required": ["location"],
    },
}


@pytest.mark.asyncio
async def test_messages_tool_result_is_error(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """A failed tool result replays through the Messages API without the provider rejecting the request."""
    if provider in (*LOCAL_PROVIDERS, LLMProvider.PERPLEXITY):
        pytest.skip(f"{provider} does not support tools, skipping")

    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")
        model_id = provider_model_map[provider]
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."},
        ]
        first = await llm.amessages(model=model_id, messages=messages, max_tokens=1024, tools=[_WEATHER_TOOL])
        assert isinstance(first, MessageResponse)
        tool_use = next((block for block in first.content if isinstance(block, ToolUseBlock)), None)
        assert tool_use is not None, f"Expected a get_weather tool call, got: {first.content}"

        messages.append(
            {"role": "assistant", "content": [block.model_dump(exclude_none=True) for block in first.content]}
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": "weather service unavailable",
                        "is_error": True,
                    }
                ],
            }
        )
        result = await llm.amessages(model=model_id, messages=messages, max_tokens=1024, tools=[_WEATHER_TOOL])
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, MessageResponse)
    assert len(result.content) >= 1
