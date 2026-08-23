from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, cast

import pytest
from openai._models import construct_type
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as OpenAIChunkChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta as OpenAIChoiceDelta,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from any_llm import AnyLLM
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.minimax.minimax import MinimaxProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.types.image import ImageGenerationParams

if TYPE_CHECKING:
    from openai._streaming import AsyncStream


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-minimax-test-123")


async def _iter_chunks(chunks: list[OpenAIChatCompletionChunk]) -> AsyncIterator[OpenAIChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    p = MinimaxProvider(api_key="sk-test")
    assert p.PROVIDER_NAME == "minimax"
    assert p.API_BASE == "https://api.minimax.io/v1"
    assert p.SUPPORTS_COMPLETION is True
    assert p.SUPPORTS_COMPLETION_STREAMING is True
    assert p.SUPPORTS_COMPLETION_REASONING is True
    assert p.SUPPORTS_EMBEDDING is False
    assert p.SUPPORTS_COMPLETION_IMAGE is False
    assert p.SUPPORTS_IMAGE_GENERATION is True
    assert p.SUPPORTS_COMPLETION_PDF is False
    assert p.SUPPORTS_LIST_MODELS is False


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    p = AnyLLM.create("minimax", api_key="sk-1")
    assert isinstance(p, MinimaxProvider)
    assert p.PROVIDER_NAME == "minimax"

    supported = AnyLLM.get_supported_providers()
    assert "minimax" in supported


def test_unsupported_response_format() -> None:
    """Test that response_format raises UnsupportedParameterError."""

    class ResponseModel(BaseModel):
        answer: str

    params = CompletionParams(
        model_id="MiniMax-M2",
        messages=[{"role": "user", "content": "Hello"}],
        response_format=ResponseModel,
    )
    with pytest.raises(UnsupportedParameterError, match="'response_format' is not supported for minimax"):
        MinimaxProvider._convert_completion_params(params)


def test_convert_completion_params_without_response_format() -> None:
    """Test that params are converted correctly when no response_format is set."""
    params = CompletionParams(
        model_id="MiniMax-M2",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
    )
    result = MinimaxProvider._convert_completion_params(params)
    assert result["temperature"] == 0.7
    assert "model_id" not in result
    assert "messages" not in result


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = MinimaxProvider.get_provider_metadata()
    assert metadata.name == "minimax"
    assert metadata.env_key == "MINIMAX_API_KEY"
    assert metadata.doc_url == "https://platform.minimax.io/docs"
    assert metadata.completion is True
    assert metadata.embedding is False
    assert metadata.image is False
    assert metadata.image_generation is True


@pytest.mark.asyncio
async def test_image_generation_uses_native_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": {"image_urls": ["https://img.example/image.png"]}, "base_resp": {"status_code": 0}}

    class FakeClient:
        def __init__(self) -> None:
            self.request: dict[str, object] | None = None

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, **kwargs: object) -> FakeResponse:
            self.request = {"url": url, **kwargs}
            return FakeResponse()

    client = FakeClient()
    monkeypatch.setattr("any_llm.providers.minimax.minimax.httpx.AsyncClient", lambda: client)
    provider = MinimaxProvider(api_key="sk-test")
    result = await provider._aimage_generation(
        ImageGenerationParams(model_id="image-01", prompt="a cat", aspect_ratio="16:9")
    )

    assert result.data[0].url == "https://img.example/image.png"
    assert client.request is not None
    assert client.request["url"] == "https://api.minimax.io/v1/image_generation"
    assert client.request["json"] == {"model": "image-01", "prompt": "a cat", "aspect_ratio": "16:9"}


@pytest.mark.asyncio
async def test_stream_preserves_usage_only_chunk() -> None:
    """Keep usage-only tail chunks while filtering unrelated empty chunks."""
    provider = MinimaxProvider(api_key="sk-test")
    chunks = [
        OpenAIChatCompletionChunk(
            id="minimax-content",
            choices=[
                OpenAIChunkChoice(
                    index=0,
                    finish_reason=None,
                    delta=OpenAIChoiceDelta(content="answer"),
                )
            ],
            created=1234567890,
            model="MiniMax-M3",
            object="chat.completion.chunk",
        ),
        OpenAIChatCompletionChunk(
            id="minimax-empty",
            choices=[],
            created=1234567890,
            model="MiniMax-M3",
            object="chat.completion.chunk",
        ),
        OpenAIChatCompletionChunk(
            id="minimax-usage",
            choices=[],
            created=1234567890,
            model="MiniMax-M3",
            object="chat.completion.chunk",
            usage=CompletionUsage(prompt_tokens=11, completion_tokens=2, total_tokens=13),
        ),
    ]
    stream = cast("AsyncStream[OpenAIChatCompletionChunk]", _iter_chunks(chunks))

    converted = provider._convert_completion_response_async(stream)

    assert not isinstance(converted, ChatCompletion)
    result: list[ChatCompletionChunk] = [chunk async for chunk in converted]
    assert len(result) == 2
    assert result[0].choices[0].delta.content == "answer"
    assert result[0].usage is None
    assert result[1].choices == []
    assert result[1].usage is not None
    assert result[1].usage.prompt_tokens == 11
    assert result[1].usage.completion_tokens == 2
    assert result[1].usage.total_tokens == 13


@pytest.mark.asyncio
async def test_stream_preserves_usage_on_chunk_without_delta() -> None:
    """Minimax attaches usage to a terminal chunk whose choice has no delta (see #657)."""
    provider = MinimaxProvider(api_key="sk-test")
    terminal = cast(
        "OpenAIChatCompletionChunk",
        construct_type(
            value={
                "id": "minimax-terminal",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "MiniMax-M3",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
            },
            type_=OpenAIChatCompletionChunk,
        ),
    )
    assert terminal.choices[0].delta is None

    stream = cast("AsyncStream[OpenAIChatCompletionChunk]", _iter_chunks([terminal]))
    converted = provider._convert_completion_response_async(stream)

    assert not isinstance(converted, ChatCompletion)
    result: list[ChatCompletionChunk] = [chunk async for chunk in converted]
    assert len(result) == 1
    assert result[0].choices == []
    assert result[0].usage is not None
    assert result[0].usage.total_tokens == 13
