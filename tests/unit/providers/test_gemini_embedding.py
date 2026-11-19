import base64
import json
import struct
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from google.genai import types
from google.genai.errors import ClientError

from any_llm import AnyLLM
from any_llm.providers.vertexai import VertexaiProvider

if TYPE_CHECKING:
    from any_llm.providers.gemini import GeminiProvider


@asynccontextmanager
async def _gemini_provider(
    handler: Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]],
) -> AsyncIterator["GeminiProvider"]:
    """Build a Developer API provider whose real SDK sends requests to ``handler``."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = cast(
            "GeminiProvider",
            AnyLLM.create(
                "gemini",
                api_key="test-api-key",
                vertexai=False,
                http_options=types.HttpOptions(httpx_async_client=client),
            ),
        )
        try:
            yield provider
        finally:
            await provider.client.aio.aclose()


@pytest.mark.parametrize("model", ["gemini-embedding-2", "gemini-embedding-001"])
@pytest.mark.parametrize("inputs", ["hello", ["hello"], ["ASCII first", '東京 "quoted"', "last café 🧪"]])
@pytest.mark.parametrize("encoding_format", ["float", "base64"])
@pytest.mark.asyncio
async def test_gemini_embedding_text_elements_reach_official_sdk_wire(
    model: str, inputs: str | list[str], encoding_format: str
) -> None:
    """Check real SDK wire boundaries separately from synthetic response values."""
    requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return one synthetic vector per request serialized by the real SDK."""
        assert request.url.path == f"/v1beta/models/{model}:batchEmbedContents"
        body = json.loads(request.content)
        requests.extend(body["requests"])
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [float(index), 0.5]} for index in range(len(body["requests"]))]},
        )

    async with _gemini_provider(handler) as provider:
        result = await provider.aembedding(
            model=model,
            inputs=inputs,
            encoding_format=encoding_format,
            config=types.EmbedContentConfig(output_dimensionality=2),
        )

    texts = [inputs] if isinstance(inputs, str) else inputs
    assert [request["content"]["parts"] for request in requests] == [[{"text": text}] for text in texts]
    assert all(request["model"] == f"models/{model}" for request in requests)
    assert all(request["outputDimensionality"] == 2 for request in requests)
    assert result.model == model
    assert [entry.index for entry in result.data] == list(range(len(texts)))
    for index, entry in enumerate(result.data):
        if encoding_format == "base64":
            assert isinstance(entry.embedding, str)
            assert struct.unpack("<2f", base64.b64decode(entry.embedding)) == (float(index), 0.5)
        else:
            assert entry.embedding == [float(index), 0.5]


@pytest.mark.asyncio
async def test_gemini_embedding_empty_batch_is_rejected_before_http() -> None:
    """Keep the SDK's empty-input validation ahead of HTTP transport."""

    async def handler(request: httpx.Request) -> httpx.Response:
        """Fail if invalid empty inputs escape the SDK's local validation."""
        pytest.fail(f"Empty batch must not make a request: {request.url}")

    async with _gemini_provider(handler) as provider:
        with pytest.raises(ValueError, match=r"contents.*required"):
            await provider.aembedding(model="gemini-embedding-2", inputs=[])


@pytest.mark.asyncio
async def test_gemini_embedding_preserves_sdk_error() -> None:
    """Keep real SDK client errors visible through the public embedding call."""

    async def handler(request: httpx.Request) -> httpx.Response:
        """Supply a service error payload to the real SDK's error conversion."""
        return httpx.Response(
            400, json={"error": {"code": 400, "message": "invalid text", "status": "INVALID_ARGUMENT"}}
        )

    async with _gemini_provider(handler) as provider:
        with pytest.raises(ClientError, match="invalid text"):
            await provider.aembedding(model="gemini-embedding-2", inputs=["hello"])


@pytest.mark.parametrize("inputs", ["hello", ["first", "second"]])
def test_vertex_embedding_keeps_original_contents(inputs: str | list[str]) -> None:
    """Protect Vertex's shared request shape from Developer API normalization."""
    config = types.EmbedContentConfig(output_dimensionality=2)
    converted = VertexaiProvider._convert_embedding_params(inputs, config=config)
    assert converted["contents"] is inputs
    assert converted["config"] is config


def test_convert_embedding_params_maps_dimensions_to_output_dimensionality() -> None:
    from any_llm.providers.gemini import GeminiProvider

    converted = GeminiProvider._convert_embedding_params("hi", dimensions=256)

    assert "dimensions" not in converted
    assert converted["config"].output_dimensionality == 256


def test_convert_embedding_params_merges_dimensions_into_existing_config() -> None:
    from any_llm.providers.gemini import GeminiProvider

    from_object = GeminiProvider._convert_embedding_params(
        "hi", dimensions=64, config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    from_dict = GeminiProvider._convert_embedding_params("hi", dimensions=64, config={"task_type": "RETRIEVAL_QUERY"})

    assert from_object["config"].output_dimensionality == 64
    assert from_object["config"].task_type == "RETRIEVAL_QUERY"
    assert from_dict["config"] == {"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 64}


def test_convert_embedding_params_without_dimensions_adds_no_config() -> None:
    from any_llm.providers.gemini import GeminiProvider

    assert GeminiProvider._convert_embedding_params("hi") == {"contents": "hi"}
