"""Integration test for Gemini cached token reporting.

Creates a real context cache via the Google GenAI SDK, issues a completion
through any-llm, and asserts that ``prompt_tokens_details.cached_tokens``
is populated in the response.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) to be set.
"""

import os
from collections.abc import AsyncIterator, Generator

import pytest
from google import genai
from google.genai import types

from any_llm import AnyLLM, LLMProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk

# Repeat text to comfortably exceed the 1,024-token caching minimum.
_LONG_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that answers questions concisely. "
    "The following is reference material that you should use to answer questions.\n\n"
) + (
    "The Gemini family of models is built by Google DeepMind. "
    "Context caching allows developers to cache frequently used input tokens "
    "so they can be reused across multiple requests without being re-processed. "
    "Cached tokens are billed at a reduced rate compared to regular input tokens. "
    "This is especially useful for long system instructions, large documents, "
    "or video and audio content that is referenced repeatedly.\n"
) * 200  # ~10,000 tokens — well above the 1,024 minimum


@pytest.fixture
def gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not set")
    return genai.Client(api_key=api_key)


@pytest.fixture
def gemini_model(provider_model_map: dict[LLMProvider, str]) -> str:
    return provider_model_map[LLMProvider.GEMINI]


@pytest.fixture
def cached_content(gemini_client: genai.Client, gemini_model: str) -> Generator[types.CachedContent]:
    cache = gemini_client.caches.create(
        model=f"models/{gemini_model}",
        config=types.CreateCachedContentConfig(
            display_name="any-llm-cached-tokens-test",
            system_instruction=_LONG_SYSTEM_INSTRUCTION,
            ttl="120s",
        ),
    )
    yield cache
    assert cache.name is not None
    gemini_client.caches.delete(name=cache.name)


@pytest.mark.asyncio
async def test_gemini_cached_tokens_non_streaming(
    cached_content: types.CachedContent,
    gemini_model: str,
) -> None:
    """Non-streaming completion reports cached_tokens in prompt_tokens_details."""
    llm = AnyLLM.create(LLMProvider.GEMINI)

    result = await llm.acompletion(
        model=gemini_model,
        messages=[{"role": "user", "content": "What is context caching? Reply in one sentence."}],
        cached_content=cached_content.name,
    )

    assert isinstance(result, ChatCompletion)
    assert result.usage is not None
    assert result.usage.prompt_tokens_details is not None, (
        "prompt_tokens_details should be set when a context cache is used"
    )
    assert result.usage.prompt_tokens_details.cached_tokens is not None
    assert result.usage.prompt_tokens_details.cached_tokens > 0


@pytest.mark.asyncio
async def test_gemini_cached_tokens_streaming(
    cached_content: types.CachedContent,
    gemini_model: str,
) -> None:
    """Streaming completion reports cached_tokens in the final chunk's usage."""
    llm = AnyLLM.create(LLMProvider.GEMINI)

    response = await llm.acompletion(
        model=gemini_model,
        messages=[{"role": "user", "content": "What is context caching? Reply in one sentence."}],
        cached_content=cached_content.name,
        stream=True,
    )

    assert isinstance(response, AsyncIterator)

    last_chunk_with_usage: ChatCompletionChunk | None = None
    async for chunk in response:
        assert isinstance(chunk, ChatCompletionChunk)
        if chunk.usage is not None:
            last_chunk_with_usage = chunk

    assert last_chunk_with_usage is not None, "At least one streaming chunk should contain usage"
    assert last_chunk_with_usage.usage is not None
    assert last_chunk_with_usage.usage.prompt_tokens_details is not None, (
        "prompt_tokens_details should be set when a context cache is used"
    )
    assert last_chunk_with_usage.usage.prompt_tokens_details.cached_tokens is not None
    assert last_chunk_with_usage.usage.prompt_tokens_details.cached_tokens > 0
