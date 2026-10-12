import base64
import struct
import sys
from operator import attrgetter
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from any_llm import AnyLLM
from any_llm.api import aembedding
from any_llm.constants import LLMProvider
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.openai.openai import OpenaiProvider
from any_llm.types.completion import CreateEmbeddingResponse, Embedding, Usage
from any_llm.utils.embedding import apply_encoding_format, pop_encoding_format


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


@pytest.mark.parametrize("encoding_format", [None, "float", "base64"])
def test_pop_encoding_format_removes_it_from_kwargs(encoding_format: str | None) -> None:
    kwargs = {"encoding_format": encoding_format, "dimensions": 3}

    assert pop_encoding_format(kwargs, "cohere") == encoding_format
    assert kwargs == {"dimensions": 3}


def test_pop_encoding_format_rejects_unknown_format() -> None:
    with pytest.raises(UnsupportedParameterError, match="encoding_format='int8'"):
        pop_encoding_format({"encoding_format": "int8"}, "cohere")


@pytest.mark.parametrize("encoding_format", [None, "float"])
def test_apply_encoding_format_keeps_floats(encoding_format: str | None) -> None:
    response = _embedding_response()

    assert apply_encoding_format(response, encoding_format) is response  # type: ignore[arg-type]


def test_apply_encoding_format_base64_matches_openai_encoding() -> None:
    result = apply_encoding_format(_embedding_response(), "base64")

    encoded = result.data[0].embedding
    assert isinstance(encoded, str)
    # OpenAI clients decode base64 embeddings as little-endian float32.
    decoded = struct.unpack("<3f", base64.b64decode(encoded))
    assert decoded == pytest.approx([0.1, 0.2, 0.3])
    assert result.model == "text-embedding-3-small"
    assert result.usage.total_tokens == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "embed_method"),
    [
        ("cohere", "client.embed"),
        ("gemini", "client.aio.models.embed_content"),
        ("ollama", "client.embed"),
        ("voyage", "client.embed"),
    ],
)
@pytest.mark.parametrize("encoding_format", ["float", "base64"])
async def test_float_only_providers_do_not_send_encoding_format(
    provider: str, embed_method: str, encoding_format: str
) -> None:
    try:
        llm = AnyLLM.create(provider, api_key="test-key")
    except ImportError as exc:
        pytest.skip(str(exc))
    owner_path, method_name = embed_method.rsplit(".", 1)
    owner = attrgetter(owner_path)(llm)

    with (
        patch.object(owner, method_name, autospec=True) as mock_embed,
        patch.object(type(llm), "_convert_embedding_response", return_value=_embedding_response()),
    ):
        result = await llm._aembedding("embed-model", "hello", encoding_format=encoding_format)

    assert "encoding_format" not in mock_embed.call_args.kwargs
    embedding = result.data[0].embedding
    if encoding_format == "float":
        assert embedding == [0.1, 0.2, 0.3]
    else:
        assert isinstance(embedding, str)
        assert struct.unpack("<3f", base64.b64decode(embedding)) == pytest.approx([0.1, 0.2, 0.3])
