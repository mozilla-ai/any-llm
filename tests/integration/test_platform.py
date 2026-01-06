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


@pytest.fixture
def any_llm_key() -> str:
    """Fixture to provide the ANY_LLM_KEY from environment."""
    key = os.getenv("ANY_LLM_KEY")
    if not key:
        pytest.skip("ANY_LLM_KEY not provided")
    return key


@pytest.fixture
def platform_provider_openai(any_llm_key: str) -> PlatformProvider:
    """Fixture to create a platform provider with OpenAI."""
    provider = AnyLLM.create(LLMProvider.OPENAI, api_key=any_llm_key)
    assert isinstance(provider, PlatformProvider)
    return provider


@pytest.mark.asyncio
async def test_platform_provider_basic_completion(platform_provider_openai: PlatformProvider) -> None:
    """Test basic completion through platform provider."""
    # Make a simple completion request
    result = await platform_provider_openai.acompletion(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Say hello"}],
    )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content is not None
    assert len(result.choices[0].message.content) > 0
    assert result.usage is not None
    assert result.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_platform_provider_streaming_completion(platform_provider_openai: PlatformProvider) -> None:
    """Test streaming completion through platform provider."""
    # Make a streaming completion request
    stream = await platform_provider_openai.acompletion(
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
async def test_platform_provider_parallel_requests(platform_provider_openai: PlatformProvider) -> None:
    """Test that platform provider can handle parallel requests."""
    import asyncio

    # Make multiple parallel requests
    results = await asyncio.gather(
        platform_provider_openai.acompletion(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        ),
        platform_provider_openai.acompletion(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "What is 3+3?"}],
        ),
        platform_provider_openai.acompletion(
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


@pytest.mark.parametrize(
    ("provider", "model", "expected_provider_name"),
    [
        (LLMProvider.OPENAI, "gpt-5-nano", "openai"),
        (LLMProvider.ANTHROPIC, "claude-3-5-haiku-latest", "anthropic"),
        (LLMProvider.GEMINI, "gemini-3-flash-preview", "gemini"),
    ],
)
@pytest.mark.asyncio
async def test_platform_provider_with_different_providers(
    any_llm_key: str,
    provider: LLMProvider,
    model: str,
    expected_provider_name: str,
) -> None:
    """Test that platform provider can wrap different providers."""
    try:
        platform_provider = AnyLLM.create(provider, api_key=any_llm_key)
        assert isinstance(platform_provider, PlatformProvider)
        assert platform_provider.provider.PROVIDER_NAME == expected_provider_name

        result = await platform_provider.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert isinstance(result, ChatCompletion)
        assert result.choices[0].message.content is not None
    except (httpx.HTTPStatusError, MissingApiKeyError):
        pytest.skip(f"{expected_provider_name} not configured in platform or API key issue")


@pytest.mark.asyncio
async def test_platform_provider_embedding(platform_provider_openai: PlatformProvider) -> None:
    """Test embedding through platform provider."""
    if not platform_provider_openai.provider.SUPPORTS_EMBEDDING:
        pytest.skip("Provider does not support embeddings")

    try:
        result = await platform_provider_openai.aembedding(
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
async def test_platform_provider_error_handling(platform_provider_openai: PlatformProvider) -> None:
    """Test that platform provider properly handles errors."""
    # Test with invalid model
    with pytest.raises((httpx.HTTPStatusError, Exception)):
        await platform_provider_openai.acompletion(
            model="invalid-model-that-does-not-exist-12345",
            messages=[{"role": "user", "content": "Hello"}],
        )


@pytest.mark.asyncio
async def test_platform_provider_list_models(platform_provider_openai: PlatformProvider) -> None:
    """Test listing models through platform provider."""
    if not platform_provider_openai.provider.SUPPORTS_LIST_MODELS:
        pytest.skip("Provider does not support list_models")

    try:
        models = await platform_provider_openai.alist_models()
        assert models is not None
        assert len(models) > 0
    except (httpx.HTTPStatusError, NotImplementedError):
        pytest.skip("List models not fully supported for this provider")


@pytest.mark.asyncio
async def test_platform_provider_with_system_message(platform_provider_openai: PlatformProvider) -> None:
    """Test platform provider with system messages."""
    result = await platform_provider_openai.acompletion(
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
