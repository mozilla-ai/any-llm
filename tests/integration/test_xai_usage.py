from collections.abc import AsyncIterator
from typing import Any

import pytest

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage, CompletionUsage
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.parametrize("provider", [LLMProvider.XAI]),
]

_MESSAGES: list[dict[str, Any] | ChatCompletionMessage] = [
    {"role": "user", "content": "Please say hello! Think before you respond."}
]


def _create_llm(provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]) -> AnyLLM:
    try:
        return AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")


def _assert_reasoning_counted_once(usage: CompletionUsage | None) -> None:
    """xAI reports reasoning either apart from completion_tokens or inside it; either way it must
    end up inside completion_tokens exactly once."""
    assert usage is not None
    assert usage.completion_tokens_details is not None
    reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
    assert reasoning_tokens is not None
    assert 0 < reasoning_tokens < usage.completion_tokens
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens is not None


async def test_xai_reasoning_tokens_count_as_completion_tokens(
    provider: LLMProvider,
    provider_reasoning_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    llm = _create_llm(provider, provider_client_config)

    result = await llm.acompletion(model=provider_reasoning_model_map[provider], messages=_MESSAGES)

    assert isinstance(result, ChatCompletion)
    _assert_reasoning_counted_once(result.usage)


async def test_xai_stream_reasoning_tokens_count_as_completion_tokens(
    provider: LLMProvider,
    provider_reasoning_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    llm = _create_llm(provider, provider_client_config)

    stream = await llm.acompletion(model=provider_reasoning_model_map[provider], messages=_MESSAGES, stream=True)

    assert isinstance(stream, AsyncIterator)
    usage = None
    async for chunk in stream:
        assert isinstance(chunk, ChatCompletionChunk)
        if chunk.usage is not None:
            usage = chunk.usage
    _assert_reasoning_counted_once(usage)
