"""prompt_cache_key reaches providers whose APIs accept it, and they accept the request."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.parametrize("provider", [LLMProvider.MISTRAL, LLMProvider.TOGETHER]),
]

_MESSAGES: list[dict[str, Any] | ChatCompletionMessage] = [{"role": "user", "content": "Say ok."}]


def _create_llm(provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]) -> AnyLLM:
    try:
        return AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")


async def test_completion_with_prompt_cache_key(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    llm = _create_llm(provider, provider_client_config)

    result = await llm.acompletion(
        model=provider_model_map[provider], messages=_MESSAGES, max_tokens=16, prompt_cache_key="any-llm-tests"
    )

    assert isinstance(result, ChatCompletion)
    assert result.choices


async def test_stream_with_prompt_cache_key(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    llm = _create_llm(provider, provider_client_config)

    stream = await llm.acompletion(
        model=provider_model_map[provider],
        messages=_MESSAGES,
        max_tokens=16,
        prompt_cache_key="any-llm-tests",
        stream=True,
    )

    assert isinstance(stream, AsyncIterator)
    chunks = [chunk async for chunk in stream]
    assert chunks
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
