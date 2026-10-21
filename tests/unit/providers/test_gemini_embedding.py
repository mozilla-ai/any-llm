import base64
import json
import struct
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from google.genai import types
from google.genai.errors import ClientError

from any_llm import AnyLLM
from any_llm.providers.vertexai import VertexaiProvider

if TYPE_CHECKING:
    from any_llm.providers.gemini import GeminiProvider


@pytest.mark.parametrize("model", ["gemini-embedding-2", "gemini-embedding-001"])
@pytest.mark.parametrize("inputs", ["hello", ["hello"], ["ASCII first", '東京 "quoted"', "last café 🧪"]])
@pytest.mark.parametrize("encoding_format", ["float", "base64"])
@pytest.mark.asyncio
async def test_gemini_embedding_text_elements_reach_official_sdk_wire(
    model: str, inputs: str | list[str], encoding_format: str
) -> None:
    requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/v1beta/models/{model}:batchEmbedContents"
        body = json.loads(request.content)
        requests.extend(body["requests"])
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [float(index), 0.5]} for index in range(len(body["requests"]))]},
        )

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
            result = await provider.aembedding(
                model=model,
                inputs=inputs,
                encoding_format=encoding_format,
                config=types.EmbedContentConfig(output_dimensionality=2),
            )
        finally:
            await provider.client.aio.aclose()

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
    async def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"Empty batch must not make a request: {request.url}")

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
            with pytest.raises(ValueError, match=r"contents.*required"):
                await provider.aembedding(model="gemini-embedding-2", inputs=[])
        finally:
            await provider.client.aio.aclose()


@pytest.mark.asyncio
async def test_gemini_embedding_preserves_sdk_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"code": 400, "message": "invalid text", "status": "INVALID_ARGUMENT"}}
        )

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
            with pytest.raises(ClientError, match="invalid text"):
                await provider.aembedding(model="gemini-embedding-2", inputs=["hello"])
        finally:
            await provider.client.aio.aclose()


@pytest.mark.parametrize("inputs", ["hello", ["first", "second"]])
def test_vertex_embedding_keeps_original_contents(inputs: str | list[str]) -> None:
    config = types.EmbedContentConfig(output_dimensionality=2)
    converted = VertexaiProvider._convert_embedding_params(inputs, config=config)
    assert converted["contents"] is inputs
    assert converted["config"] is config
