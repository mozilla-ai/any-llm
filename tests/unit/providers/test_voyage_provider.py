from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.providers.voyage import VoyageProvider


@contextmanager
def mock_voyage_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.voyage.voyage.AsyncClient") as mock_async_client,
        patch("any_llm.providers.voyage.utils._create_openai_embedding_response_from_voyage") as mock_convert_response,
    ):
        mock_convert_response.return_value = {
            "data": [
                {
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0,
                    "object": "embedding",
                }
            ],
            "model": "voyage-large-2",
            "object": "list",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        mock_client = mock_async_client.return_value
        mock_embed_result = Mock()
        mock_embed_result.embeddings = [[0.1, 0.2, 0.3]]
        mock_embed_result.total_tokens = 5
        mock_client.embed = AsyncMock(return_value=mock_embed_result)

        yield mock_async_client


@pytest.mark.asyncio
async def test_embedding_with_single_text() -> None:
    """Test that embedding works correctly with a single text input."""
    api_key = "test-api-key"
    model = "voyage-large-2"
    text = "Hello world"

    with mock_voyage_provider() as mock_async_client:
        provider = VoyageProvider(api_key=api_key)
        await provider.aembedding(model=model, inputs=text)

        mock_async_client.return_value.embed.assert_called_once()
        call_args = mock_async_client.return_value.embed.call_args
        assert call_args[1]["model"] == model
        assert call_args[1]["texts"] == [text]


@pytest.mark.asyncio
async def test_embedding_with_multiple_texts() -> None:
    """Test that embedding works correctly with multiple text inputs."""
    api_key = "test-api-key"
    model = "voyage-large-2"
    texts = ["Hello world", "How are you?", "Good morning"]

    with mock_voyage_provider() as mock_async_client:
        provider = VoyageProvider(api_key=api_key)
        await provider.aembedding(model=model, inputs=texts)

        mock_async_client.return_value.embed.assert_called_once()
        call_args = mock_async_client.return_value.embed.call_args
        assert call_args[1]["model"] == model
        assert call_args[1]["texts"] == texts


@pytest.mark.asyncio
async def test_embedding_with_additional_kwargs() -> None:
    """Test that embedding passes through additional kwargs."""
    api_key = "test-api-key"
    model = "voyage-large-2"
    text = "Hello world"
    truncation = True
    input_type = "document"

    with mock_voyage_provider() as mock_async_client:
        provider = VoyageProvider(api_key=api_key)
        await provider.aembedding(model=model, inputs=text, truncation=truncation, input_type=input_type)

        mock_async_client.return_value.embed.assert_called_once()
        call_args = mock_async_client.return_value.embed.call_args
        assert call_args[1]["model"] == model
        assert call_args[1]["texts"] == [text]
        assert call_args[1]["truncation"] == truncation
        assert call_args[1]["input_type"] == input_type


def test_convert_embedding_params_single_string() -> None:
    """Test that _convert_embedding_params correctly handles a single string."""
    params = "Hello world"
    result = VoyageProvider._convert_embedding_params(params)
    assert result == {"texts": ["Hello world"]}


def test_convert_embedding_params_list_of_strings() -> None:
    """Test that _convert_embedding_params correctly handles a list of strings."""
    params = ["Hello", "world", "test"]
    result = VoyageProvider._convert_embedding_params(params)
    assert result == {"texts": ["Hello", "world", "test"]}


def test_convert_embedding_params_with_kwargs() -> None:
    """Test that _convert_embedding_params correctly handles additional kwargs."""
    params = "Hello world"
    result = VoyageProvider._convert_embedding_params(params, truncation=True, input_type="query")
    expected = {"texts": ["Hello world"], "truncation": True, "input_type": "query"}
    assert result == expected


def test_convert_embedding_response_default_model() -> None:
    """Test that _convert_embedding_response uses default model when not provided."""
    mock_result = Mock()
    mock_result.embeddings = [[0.1, 0.2, 0.3]]
    mock_result.total_tokens = 5

    with patch("any_llm.providers.voyage.voyage._create_openai_embedding_response_from_voyage") as mock_convert:
        VoyageProvider._convert_embedding_response({"result": mock_result})
        mock_convert.assert_called_once_with("voyage-model", mock_result)
