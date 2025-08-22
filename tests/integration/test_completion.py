import asyncio
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import ProviderName, acompletion
from any_llm.exceptions import MissingApiKeyError
from any_llm.provider import ProviderFactory
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage
from tests.constants import LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_async_completion(
    provider: ProviderName,
    provider_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""
    cls = ProviderFactory.get_provider_class(provider)
    if not cls.SUPPORTS_COMPLETION:
        pytest.skip(f"{provider.value} does not support completion, skipping")

    model_id = provider_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})
    try:
        result = await acompletion(
            model=model_id,
            provider=provider,
            **extra_kwargs,
            messages=[
                {"role": "user", "content": "Hello"},
                ChatCompletionMessage(role="assistant", content="Hi!"),
                {"role": "user", "content": "What is my name?"},
            ],
        )
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content is not None
    assert hasattr(
        result.choices[0].message, "reasoning"
    )  # If all the providers are properly implementing the reasoning, this should be true


@pytest.mark.asyncio
async def test_async_completion_parallel(
    provider: ProviderName,
    provider_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    cls = ProviderFactory.get_provider_class(provider)
    if not cls.SUPPORTS_COMPLETION:
        pytest.skip(f"{provider.value} does not support completion, skipping")

    model_id = provider_model_map[provider]
    prompt_1 = "What's the capital of France?"
    prompt_2 = "What's the capital of Germany?"
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})
    try:
        results = await asyncio.gather(
            acompletion(
                model=model_id, provider=provider, **extra_kwargs, messages=[{"role": "user", "content": prompt_1}]
            ),
            acompletion(
                model=model_id, provider=provider, **extra_kwargs, messages=[{"role": "user", "content": prompt_2}]
            ),
        )
        assert isinstance(results[0], ChatCompletion)
        assert isinstance(results[1], ChatCompletion)
        assert results[0].choices[0].message.content is not None
        assert results[1].choices[0].message.content is not None
        assert "paris" in results[0].choices[0].message.content.lower()
        assert "berlin" in results[1].choices[0].message.content.lower()
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS:
            pytest.skip("Local model host is not set up, skipping")
        raise
