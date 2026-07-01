from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage
from pydantic import BaseModel

from any_llm.providers.sambanova.sambanova import SambanovaProvider
from any_llm.types.completion import CompletionParams


class PersonSchema(BaseModel):
    name: str
    age: int


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_sambanova_converts_pydantic_response_format(mock_openai_class: MagicMock) -> None:
    """Test that Pydantic BaseModel response_format is converted to JSON schema format."""
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    # Mock the response
    mock_response = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = SambanovaProvider(api_key="test-key")

    messages = [{"role": "user", "content": "Hello"}]
    params = CompletionParams(model_id="test-model", messages=messages, response_format=PersonSchema)

    await provider._acompletion(params)

    # SambaNova converts Pydantic class to dict, so .create() is used instead of .parse()
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args

    assert call_args is not None
    kwargs = call_args.kwargs

    expected_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "response_schema",
            "schema": PersonSchema.model_json_schema(),
        },
    }

    assert kwargs["response_format"] == expected_response_format
    assert kwargs["model"] == "test-model"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_sambanova_preserves_dict_response_format(mock_openai_class: MagicMock) -> None:
    """Test that dict response_format is passed through unchanged."""
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    # Mock the response
    mock_response = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = SambanovaProvider(api_key="test-key")

    messages = [{"role": "user", "content": "Hello"}]
    dict_response_format = {"type": "json_object"}
    params = CompletionParams(model_id="test-model", messages=messages, response_format=dict_response_format)

    await provider._acompletion(params)

    # Dict response_format uses .create(), not .parse()
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args

    assert call_args is not None
    kwargs = call_args.kwargs

    assert kwargs["response_format"] == dict_response_format
    assert kwargs["model"] == "test-model"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_sambanova_converts_dataclass_response_format(mock_openai_class: MagicMock) -> None:
    """Test that dataclass response_format is converted to JSON schema format."""
    from dataclasses import dataclass

    @dataclass
    class PersonDataclass:
        name: str
        age: int

    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    mock_response = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = SambanovaProvider(api_key="test-key")

    messages = [{"role": "user", "content": "Hello"}]
    params = CompletionParams(model_id="test-model", messages=messages, response_format=PersonDataclass)

    await provider._acompletion(params)

    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args

    assert call_args is not None
    kwargs = call_args.kwargs

    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "response_schema"
    assert "properties" in kwargs["response_format"]["json_schema"]["schema"]
    assert "name" in kwargs["response_format"]["json_schema"]["schema"]["properties"]
    assert "age" in kwargs["response_format"]["json_schema"]["schema"]["properties"]


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_sambanova_without_response_format(mock_openai_class: MagicMock) -> None:
    """Test normal completion without response_format."""
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    # Mock the response
    mock_response = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = SambanovaProvider(api_key="test-key")

    messages = [{"role": "user", "content": "Hello"}]
    params = CompletionParams(model_id="test-model", messages=messages)

    await provider._acompletion(params)

    # Verify the create method was called
    mock_client.chat.completions.create.assert_called_once()


def _make_embedding_response() -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")],
        model="test-model",
        object="list",
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_sambanova_embedding_omits_encoding_format(mock_openai_class: MagicMock) -> None:
    """SambaNova does not support the encoding_format parameter.

    The override should call client.post() directly instead of
    client.embeddings.create() so the SDK does not inject encoding_format.
    """
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client
    mock_client.post = AsyncMock(return_value=_make_embedding_response())

    provider = SambanovaProvider(api_key="test-key")
    result = await provider._aembedding("test-model", "Hello world")

    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    body = call_kwargs.kwargs["body"]
    assert "encoding_format" not in body
    assert body["input"] == "Hello world"
    assert body["model"] == "test-model"
    assert isinstance(result, CreateEmbeddingResponse)


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_sambanova_embedding_passes_dimensions(mock_openai_class: MagicMock) -> None:
    """The dimensions kwarg should be forwarded in the request body."""
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client
    mock_client.post = AsyncMock(return_value=_make_embedding_response())

    provider = SambanovaProvider(api_key="test-key")
    await provider._aembedding("test-model", "Hello world", dimensions=512)

    body = mock_client.post.call_args.kwargs["body"]
    assert body["dimensions"] == 512
    assert "encoding_format" not in body


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_sambanova_embedding_with_list_input(mock_openai_class: MagicMock) -> None:
    """A list of strings should be forwarded as the input value."""
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client
    mock_client.post = AsyncMock(return_value=_make_embedding_response())

    provider = SambanovaProvider(api_key="test-key")
    await provider._aembedding("test-model", ["Hello", "world"])

    body = mock_client.post.call_args.kwargs["body"]
    assert body["input"] == ["Hello", "world"]


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_sambanova_embedding_raises_when_unsupported(mock_openai_class: MagicMock) -> None:
    """_aembedding should raise NotImplementedError when SUPPORTS_EMBEDDING is False."""
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    provider = SambanovaProvider(api_key="test-key")
    provider.SUPPORTS_EMBEDDING = False

    with pytest.raises(NotImplementedError, match="does not support embeddings"):
        await provider._aembedding("test-model", "Hello world")


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_sambanova_embedding_without_dimensions(mock_openai_class: MagicMock) -> None:
    """When dimensions is not passed, the body should not include it."""
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client
    mock_client.post = AsyncMock(return_value=_make_embedding_response())

    provider = SambanovaProvider(api_key="test-key")
    await provider._aembedding("test-model", "Hello world")

    body = mock_client.post.call_args.kwargs["body"]
    assert "dimensions" not in body
    assert body["input"] == "Hello world"
    assert body["model"] == "test-model"
