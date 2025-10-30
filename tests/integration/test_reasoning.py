from collections.abc import AsyncIterable
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.anthropic.utils import DEFAULT_MAX_TOKENS
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_completion_reasoning(
    provider: LLMProvider,
    provider_reasoning_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION_REASONING:
            pytest.skip(f"{provider.value} does not support reasoning, skipping")

        model_id = provider_reasoning_model_map[provider]

        result = await llm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": "Please say hello! Think very briefly before you respond."}],
            reasoning_effort="low"
            if provider
            in (
                LLMProvider.ANTHROPIC,
                LLMProvider.GEMINI,
                LLMProvider.OLLAMA,
                LLMProvider.OPENROUTER,
                LLMProvider.VERTEXAI,
                LLMProvider.BEDROCK,
                LLMProvider.PORTKEY,
                LLMProvider.SAMBANOVA,
                LLMProvider.TOGETHER,
                LLMProvider.PORTKEY,
            )
            else "auto",
            max_tokens=4999
            if LLMProvider.FIREWORKS == provider
            else DEFAULT_MAX_TOKENS,  # Fireworks forces streaming if max_tokens is 5000+, Portkey with anthropic needed a max tokens value to be set (because it's an anthropic model)
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content is not None
    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content is not None


@pytest.mark.asyncio
async def test_completion_reasoning_streaming(
    provider: LLMProvider,
    provider_reasoning_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that reasoning works with streaming for supported providers."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION_REASONING:
            pytest.skip(f"{provider.value} does not support reasoning, skipping")
        if not llm.SUPPORTS_COMPLETION_STREAMING:
            pytest.skip(f"{provider.value} does not support streaming completion, skipping")

        model_id = provider_reasoning_model_map[provider]

        output = ""
        reasoning = ""
        num_chunks = 0
        results = await llm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": "Please say hello! Think before you respond."}],
            stream=True,
            reasoning_effort="low"
            if provider
            in (
                LLMProvider.ANTHROPIC,
                LLMProvider.GEMINI,
                LLMProvider.OLLAMA,
                LLMProvider.OPENROUTER,
                LLMProvider.VERTEXAI,
                LLMProvider.BEDROCK,
                LLMProvider.PORTKEY,
                LLMProvider.SAMBANOVA,
                LLMProvider.TOGETHER,
                LLMProvider.PORTKEY,
            )
            else "auto",
            max_tokens=4999
            if LLMProvider.FIREWORKS == provider
            else DEFAULT_MAX_TOKENS,  # Fireworks forces streaming if max_tokens is 5000+, Portkey with anthropic needed a max tokens value to be set (because it's an anthropic model)
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
        assert output != "", f"Expected non-empty output content, got {output}"

        assert reasoning != "", f"Expected non-empty reasoning content for {provider.value}, got {output}"
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
