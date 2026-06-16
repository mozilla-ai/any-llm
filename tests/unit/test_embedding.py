import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from any_llm import AnyLLM
from any_llm.api import aembedding
from any_llm.constants import LLMProvider
from any_llm.providers.openai.openai import OpenaiProvider
from any_llm.types.completion import CreateEmbeddingResponse, Embedding, Usage


def _embedding_response() -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")],
        model="text-embedding-3-small",
        object="list",
        usage=Usage(prompt_tokens=2, total_tokens=2),
    )


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
    if sys.version_info >= (3, 14) and provider.value in ("voyage", "watsonx"):
        pytest.skip(f"{provider.value} is not compatible with Python 3.14+")
    cls = AnyLLM.get_provider_class(provider)
    if not cls.SUPPORTS_EMBEDDING:
        with pytest.raises(NotImplementedError, match=None):
            await aembedding(f"{provider.value}/does-not-matter", inputs="Hello world", api_key="test_key")
    else:
        pytest.skip(f"{provider.value} supports embeddings, skipping")


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_openai_embedding_forwards_dimensions_once(mock_openai_class: MagicMock) -> None:
    """Regression: `dimensions` must reach the OpenAI client exactly once.

    It used to be forwarded both explicitly and via ``**embedding_kwargs``
    (``_convert_embedding_params`` copies it out of ``kwargs``), which raised
    ``got multiple values for keyword argument 'dimensions'`` for every
    embeddings request that set ``dimensions``.
    """
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client
    mock_response = _embedding_response()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    provider = OpenaiProvider(api_key="test_key")
    result = await provider._aembedding("text-embedding-3-small", "hello world", dimensions=512)

    assert result == mock_response
    mock_client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small", dimensions=512, input="hello world"
    )


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_openai_embedding_omits_dimensions_when_absent(mock_openai_class: MagicMock) -> None:
    """When `dimensions` is not passed, it must not be forwarded to the client."""
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client
    mock_client.embeddings.create = AsyncMock(return_value=_embedding_response())

    provider = OpenaiProvider(api_key="test_key")
    await provider._aembedding("text-embedding-3-small", "hello world")

    kwargs = mock_client.embeddings.create.await_args.kwargs
    assert "dimensions" not in kwargs
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["input"] == "hello world"
