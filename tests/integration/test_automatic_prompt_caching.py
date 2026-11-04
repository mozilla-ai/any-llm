"""Cached-token reporting for providers that cache long prompt prefixes automatically.

These providers return ``prompt_tokens_details.cached_tokens`` without any opt-in, but only
once a prefix is warm, and a request can land on a replica whose cache is cold. Each test
therefore repeats the same request a few times and asserts that one of them reports a hit.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage, CompletionUsage
from tests.constants import EXPECTED_PROVIDERS

# Models confirmed to report cache hits; the default small models do not all do so.
_CACHING_MODELS = {
    LLMProvider.CEREBRAS: "gpt-oss-120b",
    LLMProvider.MISTRAL: "mistral-small-latest",
    LLMProvider.TOGETHER: "deepseek-ai/DeepSeek-V4.1-Flash",
}
_ATTEMPTS = 4

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.parametrize("provider", list(_CACHING_MODELS)),
]


def _create_llm(provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]) -> AnyLLM:
    try:
        return AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")


def _messages() -> list[dict[str, Any] | ChatCompletionMessage]:
    # A fresh prefix per test keeps one run from riding on a cache another run warmed.
    prefix = f"Session {uuid.uuid4()}. You are a helpful assistant. " + " ".join(
        f"Fact {i}: the sky over city {i} is blue." for i in range(800)
    )
    return [{"role": "system", "content": prefix}, {"role": "user", "content": "Say ok."}]


def _cached_tokens(usage: CompletionUsage | None) -> int:
    assert usage is not None
    if usage.prompt_tokens_details is None:
        return 0
    return usage.prompt_tokens_details.cached_tokens or 0


async def test_reports_cached_tokens(
    provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]
) -> None:
    llm = _create_llm(provider, provider_client_config)
    messages = _messages()

    cached = []
    for _ in range(_ATTEMPTS):
        result = await llm.acompletion(model=_CACHING_MODELS[provider], messages=messages, max_tokens=16)
        assert isinstance(result, ChatCompletion)
        cached.append(_cached_tokens(result.usage))
        if cached[-1] > 0:
            break

    assert cached[-1] > 0, f"no cache hit in {_ATTEMPTS} identical requests: {cached}"


async def test_stream_reports_cached_tokens(
    provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]
) -> None:
    llm = _create_llm(provider, provider_client_config)
    messages = _messages()

    cached = []
    for _ in range(_ATTEMPTS):
        stream = await llm.acompletion(
            model=_CACHING_MODELS[provider],
            messages=messages,
            max_tokens=16,
            stream=True,
            stream_options={"include_usage": True},
        )
        assert isinstance(stream, AsyncIterator)
        usage = None
        async for chunk in stream:
            assert isinstance(chunk, ChatCompletionChunk)
            usage = chunk.usage or usage
        assert usage is not None, "the stream reported no usage"
        cached.append(_cached_tokens(usage))
        if cached[-1] > 0:
            break

    assert cached[-1] > 0, f"no cache hit in {_ATTEMPTS} identical streamed requests: {cached}"
