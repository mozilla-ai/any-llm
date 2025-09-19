from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import CreateEmbeddingResponse
from tests.constants import EXPECTED_PROVIDERS


@pytest.mark.asyncio
async def test_embedding_providers_async(
    provider: LLMProvider,
    embedding_provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that all embedding-supported providers can generate embeddings successfully."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_EMBEDDING:
            pytest.skip(f"{provider.value} does not support embeddings, skipping")

        model_id = embedding_provider_model_map[provider]
        result = await llm.aembedding(model=model_id, inputs="Hello world")
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
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
    if provider not in (LLMProvider.GEMINI, LLMProvider.VERTEXAI, LLMProvider.LMSTUDIO):
        assert result.usage.prompt_tokens > 0
        assert result.usage.total_tokens > 0
