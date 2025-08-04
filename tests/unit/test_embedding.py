from unittest.mock import Mock, patch

from any_llm import embedding
from any_llm.provider import ProviderName
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage


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

        mock_provider.embedding.assert_called_once_with("test-model", "Hello world", None)
        assert result == mock_embedding_response
