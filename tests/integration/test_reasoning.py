from collections.abc import AsyncIterable
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import ProviderName, acompletion
from any_llm.exceptions import MissingApiKeyError
from any_llm.provider import ProviderFactory
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from tests.constants import LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_completion_reasoning(
    provider: ProviderName,
    provider_reasoning_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""
    cls = ProviderFactory.get_provider_class(provider)
    if not cls.SUPPORTS_COMPLETION_REASONING:
        pytest.skip(f"{provider.value} does not support completion reasoning, skipping")

    model_id = provider_reasoning_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})
    if provider in (ProviderName.ANTHROPIC, ProviderName.GOOGLE, ProviderName.OLLAMA):
        extra_kwargs["reasoning_effort"] = "low"

    try:
        result = await acompletion(
            model=model_id,
            provider=provider,
            **extra_kwargs,
            messages=[{"role": "user", "content": "Please say hello! Think very briefly before you respond."}],
        )
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content is not None
    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content is not None


@pytest.mark.asyncio
async def test_completion_reasoning_streaming(
    provider: ProviderName,
    provider_reasoning_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    """Test that reasoning works with streaming for supported providers."""
    cls = ProviderFactory.get_provider_class(provider)
    if not cls.SUPPORTS_COMPLETION_REASONING:
        pytest.skip(f"{provider.value} does not support completion reasoning, skipping")
    if not cls.SUPPORTS_COMPLETION_STREAMING:
        pytest.skip(f"{provider.value} does not support streaming completion, skipping")

    model_id = provider_reasoning_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})
    if provider in (ProviderName.ANTHROPIC, ProviderName.GOOGLE, ProviderName.OLLAMA):
        extra_kwargs["reasoning_effort"] = "low"

    try:
        output = ""
        reasoning = ""
        num_chunks = 0
        results = await acompletion(
            model=model_id,
            provider=provider,
            **extra_kwargs,
            messages=[{"role": "user", "content": "Please say hello! Think very briefly before you respond."}],
            stream=True,
        )
        assert isinstance(results, AsyncIterable)
        async for result in results:
            num_chunks += 1
            assert isinstance(result, ChatCompletionChunk)
            if len(result.choices) > 0:
                output += result.choices[0].delta.content or ""
                if result.choices[0].delta.reasoning:
                    reasoning += result.choices[0].delta.reasoning.content or ""

        assert num_chunks >= 1, f"Expected at least 1 chunk, got {num_chunks}"
        assert output.strip() != "", "Expected non-empty output content"

        assert reasoning.strip() != "", f"Expected non-empty reasoning content for {provider.value}"
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
