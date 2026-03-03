from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.messages import MessageResponse, MessageStreamEvent
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
