import httpx
import pytest
from any_llm import embedding, ProviderName
from any_llm.exceptions import MissingApiKeyError
from any_llm.provider import ProviderFactory
from openai.types import CreateEmbeddingResponse


def test_embedding_providers(provider: ProviderName, embedding_provider_model_map: dict[ProviderName, str]) -> None:
    """Test that all embedding-supported providers can generate embeddings successfully."""
    # first check if the provider supports embeddings
    providers_metadata = ProviderFactory.get_all_provider_metadata()
    provider_metadata = [metadata for metadata in providers_metadata if metadata["provider_key"] == provider.value][0]
    if not provider_metadata["embedding"]:
        pytest.skip(f"{provider.value} does not support embeddings, skipping")

    model_id = embedding_provider_model_map[provider]
    try:
        result = embedding(f"{provider.value}/{model_id}", "Hello world")
        # Verify result is a list of floats
        assert isinstance(result, CreateEmbeddingResponse)
        assert len(result.data) > 0
        assert all(isinstance(x.embedding, list) for x in result.data)
        assert result.usage.prompt_tokens > 0
        assert result.usage.total_tokens > 0
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError):
        pytest.skip(f"{provider.value} connection failed, skipping")
    except Exception as e:
        # Skip if model doesn't exist or embedding isn't actually supported
        if "model" in str(e).lower() or "embedding" in str(e).lower():
            pytest.skip(f"{provider.value} embedding model not available: {e}")
        raise
