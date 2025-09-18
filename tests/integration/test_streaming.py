from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.types.completion import ChatCompletionChunk
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_streaming_completion_async(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that streaming completion works for supported providers."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION_STREAMING:
            pytest.skip(f"{provider.value} does not support streaming completion")
        model_id = provider_model_map[provider]
        output = ""
        reasoning = ""
        num_chunks = 0
        stream = await llm.acompletion(
            model=model_id,
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
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except UnsupportedParameterError:
        pytest.skip(f"Streaming is not supported for {provider.value}")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
