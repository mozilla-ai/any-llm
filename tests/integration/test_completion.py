import asyncio
import httpx
import pytest
from any_llm import completion, acompletion, ProviderName
from any_llm.exceptions import MissingApiKeyError


def test_providers(provider: ProviderName, provider_model_map: dict[ProviderName, str]) -> None:
    """Test that all supported providers can be loaded successfully."""
    model_id = provider_model_map[provider]
    try:
        result = completion(f"{provider.value}/{model_id}", messages=[{"role": "user", "content": "Hello"}])
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError):
        if provider == ProviderName.OLLAMA:
            pytest.skip("Ollama is not set up, skipping")
        raise
    assert result.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_parallel_async_completion(provider: ProviderName, provider_model_map: dict[ProviderName, str]) -> None:
    """Test that parallel completion works."""
    model_id = provider_model_map[provider]
    prompt_1 = "What's the capital of France?"
    prompt_2 = "What's the capital of Germany?"
    try:
        results = await asyncio.gather(
            acompletion(f"{provider.value}/{model_id}", messages=[{"role": "user", "content": prompt_1}]),
            acompletion(f"{provider.value}/{model_id}", messages=[{"role": "user", "content": prompt_2}]),
        )
        assert results[0].choices[0].message.content is not None
        assert results[1].choices[0].message.content is not None
        assert "paris" in results[0].choices[0].message.content.lower()
        assert "berlin" in results[1].choices[0].message.content.lower()
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError):
        if provider == ProviderName.OLLAMA:
            pytest.skip("Ollama is not set up, skipping")
        raise
