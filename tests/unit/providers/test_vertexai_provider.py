import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from google.genai import types
from openai import AsyncOpenAI

from any_llm.providers.gemini.base import GoogleProvider
from any_llm.providers.vertexai import VertexaiProvider
from any_llm.providers.vertexai.vertexai import _is_partner_model, _partner_api_base, _VertexaiPartnerProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams

QWEN_MODEL = "qwen/qwen3-235b-a22b-instruct-2507-maas"
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


@pytest.fixture
def genai_client() -> Iterator[MagicMock]:
    with patch("any_llm.providers.vertexai.vertexai.genai.Client") as mock_client:
        api_client = mock_client.return_value._api_client
        api_client.project = "my-project"
        api_client.location = "us-south1"
        api_client._async_access_token = AsyncMock(side_effect=["token-1", "token-2"])
        yield mock_client


def _patch_openai_transport(handler: Callable[[httpx.Request], httpx.Response]) -> Any:
    def make_client(**kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return AsyncOpenAI(**kwargs)

    return patch("any_llm.providers.vertexai.vertexai.AsyncOpenAI", side_effect=make_client)


def _completion_body(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def _sse_body(model: str) -> bytes:
    chunks = [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]
    events = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return f"{events}data: [DONE]\n\n".encode()


def test_vertexai_initialization_without_api_key() -> None:
    """Test that the VertexaiProvider initializes correctly without API Key."""
    with patch("any_llm.providers.vertexai.vertexai.genai.Client"):
        provider = VertexaiProvider()
        assert provider.client is not None


def test_vertexai_timeout_in_client_args_routed_to_http_options() -> None:
    """Test that timeout in client_args is converted to HttpOptions at client init."""
    with patch("any_llm.providers.vertexai.vertexai.genai.Client") as mock_client:
        VertexaiProvider(timeout=30.0)
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "http_options" in call_kwargs
        assert call_kwargs["http_options"].timeout == 30_000


def test_vertexai_timeout_does_not_override_explicit_http_options() -> None:
    """Test that explicit http_options timeout takes precedence over client_args timeout."""
    with patch("any_llm.providers.vertexai.vertexai.genai.Client") as mock_client:
        VertexaiProvider(timeout=30.0, http_options=types.HttpOptions(timeout=10_000))
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert call_kwargs["http_options"].timeout == 10_000


def test_vertexai_completion_params_include_image_parts() -> None:
    params = CompletionParams(
        model_id="gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ],
            }
        ],
    )

    converted = VertexaiProvider._convert_completion_params(params, provider_name="vertexai")
    parts = converted["contents"][0].parts

    assert parts[0].text == "Describe this image"
    assert parts[1].file_data is not None
    assert parts[1].file_data.file_uri == "https://example.com/a.png"
    assert parts[1].file_data.mime_type == "image/png"


@pytest.mark.parametrize(
    "model_id",
    [
        QWEN_MODEL,
        "openai/gpt-oss-120b-maas",
        "deepseek-ai/deepseek-v3.1-maas",
        "llama-3.3-70b-instruct-maas",
        "meta/llama-4-maverick-17b-128e-instruct-maas",
        "minimaxai/minimax-m2-maas",
        "moonshotai/kimi-k2-thinking-maas",
        "zai-org/glm-4.7-maas",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-MAAS",
    ],
)
def test_vertexai_partner_models_are_detected(model_id: str) -> None:
    assert _is_partner_model(model_id)


@pytest.mark.parametrize(
    "model_id",
    ["gemini-2.5-flash", "publishers/google/models/gemini-2.5-pro", "openai/gpt-4o", "meta/other-model"],
)
def test_vertexai_non_partner_models_are_not_detected(model_id: str) -> None:
    assert not _is_partner_model(model_id)


def test_vertexai_partner_api_base_global_location() -> None:
    assert (
        _partner_api_base("my-project", "global")
        == "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/endpoints/openapi"
    )


def test_vertexai_partner_api_base_regional_location() -> None:
    assert (
        _partner_api_base("my-project", "us-south1")
        == "https://us-south1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-south1/endpoints/openapi"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", [QWEN_MODEL, "openai/gpt-oss-20b-maas", "zai-org/glm-4.7-maas"])
async def test_vertexai_partner_model_routes_to_openai_endpoint(genai_client: MagicMock, model_id: str) -> None:
    provider = VertexaiProvider()
    params = CompletionParams(model_id=model_id, messages=[{"role": "user", "content": "Hi"}])
    expected = MagicMock(spec=ChatCompletion)

    with (
        patch.object(_VertexaiPartnerProvider, "_acompletion", AsyncMock(return_value=expected)) as partner,
        patch.object(GoogleProvider, "_acompletion", AsyncMock()) as gemini,
    ):
        result = await provider._acompletion(params, timeout=5.0)

    assert result is expected
    partner.assert_awaited_once_with(params, timeout=5.0)
    gemini.assert_not_awaited()


@pytest.mark.asyncio
async def test_vertexai_gemini_model_keeps_generate_content_path(genai_client: MagicMock) -> None:
    provider = VertexaiProvider()
    params = CompletionParams(model_id="gemini-2.5-flash", messages=[{"role": "user", "content": "Hi"}])
    expected = MagicMock(spec=ChatCompletion)

    with (
        patch.object(_VertexaiPartnerProvider, "_acompletion", AsyncMock()) as partner,
        patch.object(GoogleProvider, "_acompletion", AsyncMock(return_value=expected)) as gemini,
    ):
        result = await provider._acompletion(params)

    assert result is expected
    gemini.assert_awaited_once_with(params)
    partner.assert_not_awaited()
    assert provider._partner_provider is None


def test_vertexai_partner_provider_is_reused(genai_client: MagicMock) -> None:
    provider = VertexaiProvider()

    first = provider._get_partner_provider()

    assert provider._get_partner_provider() is first
    assert str(first.client.base_url) == (
        "https://us-south1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-south1/endpoints/openapi/"
    )


def test_vertexai_partner_provider_uses_global_host(genai_client: MagicMock) -> None:
    genai_client.return_value._api_client.location = "global"
    provider = VertexaiProvider()

    assert str(provider._get_partner_provider().client.base_url) == (
        "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/endpoints/openapi/"
    )


@pytest.mark.asyncio
async def test_vertexai_partner_non_streaming_completion(genai_client: MagicMock) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_completion_body(QWEN_MODEL))

    params = CompletionParams(
        model_id=QWEN_MODEL,
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[WEATHER_TOOL],
        max_tokens=64,
    )
    with _patch_openai_transport(handler):
        provider = VertexaiProvider()
        result = await provider._acompletion(params)

    assert isinstance(result, ChatCompletion)
    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.name == "get_weather"  # type: ignore[union-attr]
    assert result.usage is not None
    assert result.usage.total_tokens == 12

    request = requests[0]
    assert str(request.url) == (
        "https://us-south1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-south1"
        "/endpoints/openapi/chat/completions"
    )
    assert request.headers["Authorization"] == "Bearer token-1"
    body = json.loads(request.content)
    assert body["model"] == QWEN_MODEL
    assert body["tools"] == [WEATHER_TOOL]
    assert body["max_completion_tokens"] == 64
    assert "stream" not in body


@pytest.mark.asyncio
async def test_vertexai_partner_streaming_completion_with_tools(genai_client: MagicMock) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_body(QWEN_MODEL))

    params = CompletionParams(
        model_id=QWEN_MODEL,
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[WEATHER_TOOL],
        stream=True,
    )
    with _patch_openai_transport(handler):
        provider = VertexaiProvider()
        result = await provider._acompletion(params)
        assert isinstance(result, AsyncIterator)
        chunks: list[ChatCompletionChunk] = [chunk async for chunk in result]

    assert len(chunks) == 2
    delta_tool_calls = chunks[0].choices[0].delta.tool_calls
    assert delta_tool_calls is not None
    assert delta_tool_calls[0].function is not None
    assert delta_tool_calls[0].function.name == "get_weather"
    assert chunks[1].choices[0].finish_reason == "tool_calls"

    body = json.loads(requests[0].content)
    assert body["stream"] is True
    assert body["tools"] == [WEATHER_TOOL]
    assert requests[0].headers["Authorization"] == "Bearer token-1"


@pytest.mark.asyncio
async def test_vertexai_partner_requests_fetch_a_fresh_token_each_time(genai_client: MagicMock) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_completion_body(QWEN_MODEL))

    params = CompletionParams(model_id=QWEN_MODEL, messages=[{"role": "user", "content": "Hi"}])
    with _patch_openai_transport(handler):
        provider = VertexaiProvider()
        await provider._acompletion(params)
        await provider._acompletion(params)

    assert [request.headers["Authorization"] for request in requests] == ["Bearer token-1", "Bearer token-2"]
    assert genai_client.return_value._api_client._async_access_token.await_count == 2


@pytest.mark.asyncio
async def test_vertexai_access_token_comes_from_genai_credentials(genai_client: MagicMock) -> None:
    provider = VertexaiProvider()

    assert await provider._access_token() == "token-1"
    genai_client.return_value._api_client._async_access_token.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_vertexai_partner_token_error_propagates(genai_client: MagicMock) -> None:
    genai_client.return_value._api_client._async_access_token = AsyncMock(
        side_effect=RuntimeError("Could not resolve API token from the environment")
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion_body(QWEN_MODEL))

    params = CompletionParams(model_id=QWEN_MODEL, messages=[{"role": "user", "content": "Hi"}])
    with _patch_openai_transport(handler):
        provider = VertexaiProvider()
        with pytest.raises(RuntimeError, match="Could not resolve API token"):
            await provider._acompletion(params)
