from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import ProviderName, aembedding
from any_llm.exceptions import MissingApiKeyError
from any_llm.provider import ProviderFactory
from any_llm.types.completion import CreateEmbeddingResponse


@pytest.mark.asyncio
async def test_embedding_providers_async(
    provider: ProviderName,
    embedding_provider_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    """Test that all embedding-supported providers can generate embeddings successfully."""
    cls = ProviderFactory.get_provider_class(provider)
    if not cls.SUPPORTS_EMBEDDING:
        pytest.skip(f"{provider.value} does not support embeddings, skipping")

    model_id = embedding_provider_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})
    try:
        result = await aembedding(model=model_id, provider=provider, **extra_kwargs, inputs="Hello world")
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
    if provider not in (ProviderName.GOOGLE, ProviderName.LMSTUDIO):
        assert result.usage.prompt_tokens > 0
        assert result.usage.total_tokens > 0
