from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm import AnyLLM
from any_llm.api import aembedding
from any_llm.constants import LLMProvider
from any_llm.types.completion import CreateEmbeddingResponse, Embedding, Usage


@pytest.mark.asyncio
async def test_embedding_with_api_config() -> None:
    """Test embedding works with API configuration parameters."""
    mock_provider = Mock()
    mock_embedding_response = CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")],
        model="test-model",
        object="list",
        usage=Usage(prompt_tokens=2, total_tokens=2),
    )
    mock_provider._aembedding = AsyncMock(return_value=mock_embedding_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = await aembedding(
            "openai/test-model", inputs="Hello world", api_key="test_key", api_base="https://test.example.com"
        )

        call_args = mock_create.call_args
        assert call_args[0][0] == LLMProvider.OPENAI
        assert call_args[1]["api_key"] == "test_key"
        assert call_args[1]["api_base"] == "https://test.example.com"

        mock_provider._aembedding.assert_called_once_with("test-model", "Hello world")
        assert result == mock_embedding_response


@pytest.mark.asyncio
async def test_embedding_unsupported_provider_raises_not_implemented(provider: LLMProvider) -> None:
    """Test that calling embedding on a provider that doesn't support it raises NotImplementedError."""
    cls = AnyLLM.get_provider_class(provider)
    if not cls.SUPPORTS_EMBEDDING:
        with pytest.raises(NotImplementedError, match=None):
            await aembedding(f"{provider.value}/does-not-matter", inputs="Hello world", api_key="test_key")
    else:
        pytest.skip(f"{provider.value} supports embeddings, skipping")
