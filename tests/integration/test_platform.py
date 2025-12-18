"""Integration tests for the Platform provider.

These tests verify that the Platform provider correctly wraps other providers
and tracks usage metrics when using the AnyLLM platform.
"""

import os

import httpx
import pytest

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.platform import PlatformProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk

# Skip all tests if ANY_LLM_KEY is not provided
pytestmark = pytest.mark.skipif(
    not os.getenv("ANY_LLM_KEY"),
    reason="ANY_LLM_KEY not provided - platform tests require platform credentials",
)


@pytest.mark.asyncio
async def test_platform_provider_basic_completion() -> None:
    """Test basic completion through platform provider."""
    any_llm_key = os.getenv("ANY_LLM_KEY")
    if not any_llm_key:
        pytest.skip("ANY_LLM_KEY not provided")

    # Create platform provider
    platform_provider = AnyLLM.create(LLMProvider.OPENAI, api_key=any_llm_key)

    # Verify it's a platform provider
    assert isinstance(platform_provider, PlatformProvider)

    # Make a simple completion request
    result = await platform_provider.acompletion(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Say hello"}],
    )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content is not None
    assert len(result.choices[0].message.content) > 0
    assert result.usage is not None
    assert result.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_platform_provider_streaming_completion() -> None:
    """Test streaming completion through platform provider."""
    any_llm_key = os.getenv("ANY_LLM_KEY")
    if not any_llm_key:
        pytest.skip("ANY_LLM_KEY not provided")

    platform_provider = AnyLLM.create(LLMProvider.OPENAI, api_key=any_llm_key)
    assert isinstance(platform_provider, PlatformProvider)

    # Make a streaming completion request
    stream = await platform_provider.acompletion(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    chunks_collected = []
    async for chunk in stream:  # type: ignore[union-attr]
        assert isinstance(chunk, ChatCompletionChunk)
        chunks_collected.append(chunk)

    # Verify we got multiple chunks
    assert len(chunks_collected) > 1

    # Verify the last chunk has usage info
    last_chunk = chunks_collected[-1]
    assert last_chunk.usage is not None
    assert last_chunk.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_platform_provider_parallel_requests() -> None:
    """Test that platform provider can handle parallel requests."""
    any_llm_key = os.getenv("ANY_LLM_KEY")
    if not any_llm_key:
        pytest.skip("ANY_LLM_KEY not provided")

    import asyncio

    platform_provider = AnyLLM.create(LLMProvider.OPENAI, api_key=any_llm_key)
    assert isinstance(platform_provider, PlatformProvider)

    # Make multiple parallel requests
    results = await asyncio.gather(
        platform_provider.acompletion(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        ),
        platform_provider.acompletion(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "What is 3+3?"}],
        ),
        platform_provider.acompletion(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "What is 4+4?"}],
        ),
    )

    # Verify all requests succeeded
    assert len(results) == 3
    for result in results:
        assert isinstance(result, ChatCompletion)
        assert result.choices[0].message.content is not None
        assert result.usage is not None


@pytest.mark.asyncio
async def test_platform_provider_with_anthropic() -> None:
    """Test platform provider wrapping Anthropic provider."""
    any_llm_key = os.getenv("ANY_LLM_KEY")
    if not any_llm_key:
        pytest.skip("ANY_LLM_KEY not provided")

    try:
        platform_provider = AnyLLM.create(LLMProvider.ANTHROPIC, api_key=any_llm_key)
        assert isinstance(platform_provider, PlatformProvider)

        result = await platform_provider.acompletion(
            model="claude-3-5-haiku-latest",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert isinstance(result, ChatCompletion)
        assert result.choices[0].message.content is not None
    except (httpx.HTTPStatusError, MissingApiKeyError):
        pytest.skip("Anthropic not configured in platform or API key issue")


@pytest.mark.asyncio
async def test_platform_provider_with_different_providers() -> None:
    """Test that platform provider can wrap different providers in sequence."""
    any_llm_key = os.getenv("ANY_LLM_KEY")
    if not any_llm_key:
        pytest.skip("ANY_LLM_KEY not provided")

    # Test with OpenAI
    openai_provider = AnyLLM.create(LLMProvider.OPENAI, api_key=any_llm_key)
    assert isinstance(openai_provider, PlatformProvider)
    assert openai_provider.provider.PROVIDER_NAME == "openai"

    result = await openai_provider.acompletion(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert isinstance(result, ChatCompletion)

    # Test with another provider (if configured)
    try:
        gemini_provider = AnyLLM.create(LLMProvider.GEMINI, api_key=any_llm_key)
        assert isinstance(gemini_provider, PlatformProvider)
        assert gemini_provider.provider.PROVIDER_NAME == "gemini"

        result = await gemini_provider.acompletion(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert isinstance(result, ChatCompletion)
    except (httpx.HTTPStatusError, MissingApiKeyError):
        pytest.skip("Gemini not configured in platform")


@pytest.mark.asyncio
async def test_platform_provider_embedding() -> None:
    """Test embedding through platform provider."""
    any_llm_key = os.getenv("ANY_LLM_KEY")
    if not any_llm_key:
        pytest.skip("ANY_LLM_KEY not provided")

    platform_provider = AnyLLM.create(LLMProvider.OPENAI, api_key=any_llm_key)
    assert isinstance(platform_provider, PlatformProvider)

    if not platform_provider.provider.SUPPORTS_EMBEDDING:
        pytest.skip("Provider does not support embeddings")

    try:
        result = await platform_provider.aembedding(
            model="text-embedding-ada-002",
            inputs="Hello world",
        )

        assert result.data is not None
        assert len(result.data) > 0
        assert len(result.data[0].embedding) > 0
        assert result.usage is not None
    except httpx.HTTPStatusError:
        pytest.skip("Embedding model not configured in platform")


@pytest.mark.asyncio
async def test_platform_provider_error_handling() -> None:
    """Test that platform provider properly handles errors."""
    any_llm_key = os.getenv("ANY_LLM_KEY")
    if not any_llm_key:
        pytest.skip("ANY_LLM_KEY not provided")

    platform_provider = AnyLLM.create(LLMProvider.OPENAI, api_key=any_llm_key)
    assert isinstance(platform_provider, PlatformProvider)

    # Test with invalid model
    with pytest.raises((httpx.HTTPStatusError, Exception)):
        await platform_provider.acompletion(
            model="invalid-model-that-does-not-exist-12345",
            messages=[{"role": "user", "content": "Hello"}],
        )


@pytest.mark.asyncio
async def test_platform_provider_list_models() -> None:
    """Test listing models through platform provider."""
    any_llm_key = os.getenv("ANY_LLM_KEY")
    if not any_llm_key:
        pytest.skip("ANY_LLM_KEY not provided")

    platform_provider = AnyLLM.create(LLMProvider.OPENAI, api_key=any_llm_key)
    assert isinstance(platform_provider, PlatformProvider)

    if not platform_provider.provider.SUPPORTS_LIST_MODELS:
        pytest.skip("Provider does not support list_models")

    try:
        models = await platform_provider.alist_models()
        assert models is not None
        assert len(models) > 0
    except (httpx.HTTPStatusError, NotImplementedError):
        pytest.skip("List models not fully supported for this provider")


@pytest.mark.asyncio
async def test_platform_provider_with_system_message() -> None:
    """Test platform provider with system messages."""
    any_llm_key = os.getenv("ANY_LLM_KEY")
    if not any_llm_key:
        pytest.skip("ANY_LLM_KEY not provided")

    platform_provider = AnyLLM.create(LLMProvider.OPENAI, api_key=any_llm_key)
    assert isinstance(platform_provider, PlatformProvider)

    result = await platform_provider.acompletion(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that always responds in haiku."},
            {"role": "user", "content": "Tell me about the ocean."},
        ],
    )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content is not None
    assert len(result.choices[0].message.content) > 0


@pytest.mark.asyncio
async def test_platform_provider_requires_valid_key() -> None:
    """Test that platform provider requires a valid platform key."""
    # Test with invalid key format
    with pytest.raises((ValueError, Exception)):
        AnyLLM.create(LLMProvider.OPENAI, api_key="invalid-key-format")

    # Test with valid format but likely invalid key
    invalid_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="
    platform_provider = AnyLLM.create(LLMProvider.OPENAI, api_key=invalid_key)

    # Should fail when trying to make a request
    with pytest.raises((httpx.HTTPStatusError, Exception)):
        await platform_provider.acompletion(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Hello"}],
        )
