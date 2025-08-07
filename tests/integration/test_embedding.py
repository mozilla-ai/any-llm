from typing import Any
import httpx
import pytest
from any_llm import embedding, ProviderName
from any_llm.exceptions import MissingApiKeyError
from any_llm.provider import ProviderFactory
from any_llm.types.completion import CreateEmbeddingResponse
from openai import APIConnectionError


def test_embedding_providers(
    provider: ProviderName,
    embedding_provider_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    """Test that all embedding-supported providers can generate embeddings successfully."""
    # first check if the provider supports embeddings
    providers_metadata = ProviderFactory.get_all_provider_metadata()
    provider_metadata = [metadata for metadata in providers_metadata if metadata["provider_key"] == provider.value][0]
    if not provider_metadata["embedding"]:
        pytest.skip(f"{provider.value} does not support embeddings, skipping")

    model_id = embedding_provider_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})
    try:
        result = embedding(f"{provider.value}/{model_id}", **extra_kwargs, inputs="Hello world")
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        pytest.skip(f"{provider.value} connection failed, skipping")
    except Exception as e:
        # Skip if model doesn't exist or embedding isn't actually supported
        if "model" in str(e).lower() or "embedding" in str(e).lower():
            pytest.skip(f"{provider.value} embedding model not available: {e}")
        raise
    assert isinstance(result, CreateEmbeddingResponse)
    assert len(result.data) > 0
    for entry in result.data:
        assert all(isinstance(v, float) for v in entry.embedding)
    # These providers don't output token use
    if provider not in (ProviderName.GOOGLE, ProviderName.LMSTUDIO):
        assert result.usage.prompt_tokens > 0
        assert result.usage.total_tokens > 0
