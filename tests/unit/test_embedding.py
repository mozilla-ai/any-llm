from unittest.mock import Mock, patch
import pytest

from any_llm import embedding
from any_llm.provider import ProviderFactory, ProviderName
from any_llm.types.completion import CreateEmbeddingResponse, Embedding, Usage


def test_embedding_with_api_config() -> None:
    """Test embedding works with API configuration parameters."""
    mock_provider = Mock()
    mock_embedding_response = CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")],
        model="test-model",
        object="list",
        usage=Usage(prompt_tokens=2, total_tokens=2),
    )
    mock_provider.embedding.return_value = mock_embedding_response

    with patch("any_llm.api.ProviderFactory") as mock_factory:
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "test-model")
        mock_factory.create_provider.return_value = mock_provider

        result = embedding(
            "openai/test-model", inputs="Hello world", api_key="test_key", api_base="https://test.example.com"
        )

        # Verify provider was created with correct config
        call_args = mock_factory.create_provider.call_args
        assert call_args[0][0] == ProviderName.OPENAI
        assert call_args[0][1].api_key == "test_key"
        assert call_args[0][1].api_base == "https://test.example.com"

        mock_provider.embedding.assert_called_once_with("test-model", "Hello world")
        assert result == mock_embedding_response


def test_embedding_unsupported_provider_raises_not_implemented(provider: ProviderName) -> None:
    """Test that calling embedding on a provider that doesn't support it raises NotImplementedError."""
    providers_metadata = ProviderFactory.get_all_provider_metadata()
    provider_metadata = [metadata for metadata in providers_metadata if metadata["provider_key"] == provider.value][0]
    if bool(provider_metadata.get("embedding", False)) is False:
        with pytest.raises(NotImplementedError, match=None):
            embedding(f"{provider.value}/does-not-matter", inputs="Hello world", api_key="test_key")
    else:
        pytest.skip(f"{provider.value} supports embeddings, skipping")
