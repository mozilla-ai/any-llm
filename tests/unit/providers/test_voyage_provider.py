import sys
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.providers.voyage import VoyageProvider

_SKIP_PYTHON_314 = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="voyageai is not compatible with Python 3.14+ (pydantic v1 breaking changes)",
)


@_SKIP_PYTHON_314
@pytest.mark.parametrize(
    ("api_base", "base_url"),
    [
        (None, None),
        (None, "https://sdk.example/v1"),
        ("https://proxy.example/v1", None),
        ("https://proxy.example/v1", "https://sdk.example/v1"),
    ],
)
def test_init_client_forwards_api_base(
    api_base: str | None, base_url: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("voyageai")
    from voyageai.client_async import AsyncClient

    monkeypatch.delenv("VOYAGE_API_BASE", raising=False)
    kwargs: dict[str, Any] = {"timeout": 7, "max_retries": 2}
    if base_url is not None:
        kwargs["base_url"] = base_url

    with patch("any_llm.providers.voyage.voyage.AsyncClient", wraps=AsyncClient) as client:
        provider = VoyageProvider(api_key="test-api-key", api_base=api_base, **kwargs)

    expected_kwargs = dict(kwargs)
    if api_base is not None:
        expected_kwargs["base_url"] = api_base
    client.assert_called_once_with(api_key="test-api-key", **expected_kwargs)
    assert isinstance(provider.client, AsyncClient)
    assert provider.client._params == AsyncClient(api_key="test-api-key", **expected_kwargs)._params
    assert kwargs.get("base_url") == base_url


@_SKIP_PYTHON_314
def test_init_client_forwards_api_base_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("voyageai")
    from voyageai.client_async import AsyncClient

    monkeypatch.setenv("VOYAGE_API_BASE", "https://proxy.example/v1")
    with patch("any_llm.providers.voyage.voyage.AsyncClient", wraps=AsyncClient) as client:
        provider = VoyageProvider(api_key="test-api-key")

    client.assert_called_once_with(api_key="test-api-key", base_url="https://proxy.example/v1")
    assert provider.client._params["base_url"] == "https://proxy.example/v1"


@contextmanager
def mock_voyage_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.voyage.voyage.AsyncClient", create=True) as mock_async_client,
        patch(
            "any_llm.providers.voyage.utils._create_openai_embedding_response_from_voyage",
            create=True,
        ) as mock_convert_response,
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


@_SKIP_PYTHON_314
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


@_SKIP_PYTHON_314
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


@_SKIP_PYTHON_314
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


@_SKIP_PYTHON_314
def test_convert_embedding_response_default_model() -> None:
    """Test that _convert_embedding_response uses default model when not provided."""
    mock_result = Mock()
    mock_result.embeddings = [[0.1, 0.2, 0.3]]
    mock_result.total_tokens = 5

    with patch(
        "any_llm.providers.voyage.voyage._create_openai_embedding_response_from_voyage",
        create=True,
    ) as mock_convert:
        VoyageProvider._convert_embedding_response({"result": mock_result})
        mock_convert.assert_called_once_with("voyage-model", mock_result)
