from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import ProviderName, acompletion
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.provider import ProviderFactory
from any_llm.types.completion import ChatCompletionChunk
from tests.constants import LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_streaming_completion_async(
    provider: ProviderName,
    provider_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    """Test that streaming completion works for supported providers."""
    cls = ProviderFactory.get_provider_class(provider)
    if not cls.SUPPORTS_COMPLETION_STREAMING:
        pytest.skip(f"{provider.value} does not support streaming completion")
    model_id = provider_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})
    try:
        output = ""
        reasoning = ""
        num_chunks = 0
        stream = await acompletion(
            model=model_id,
            provider=provider,
            **extra_kwargs,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that exactly follows the user request."},
                {"role": "user", "content": "Say the exact phrase:'Hello World' with no fancy formatting"},
            ],
            stream=True,
        )

        if isinstance(stream, AsyncIterator):
            async for result in stream:
                num_chunks += 1
                assert isinstance(result, ChatCompletionChunk)
                if len(result.choices) > 0:
                    output += result.choices[0].delta.content or ""
                    if result.choices[0].delta.reasoning:
                        reasoning += result.choices[0].delta.reasoning.content or ""
            assert num_chunks >= 1, f"Expected at least 1 chunk, got {num_chunks}"
            assert "hello world" in output.lower()
        else:
            msg = f"Expected AsyncIterator[ChatCompletionChunk], not {type(stream)}"
            raise TypeError(msg)
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except UnsupportedParameterError:
        pytest.skip(f"Streaming is not supported for {provider.value}")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
