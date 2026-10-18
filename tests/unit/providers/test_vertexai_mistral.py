import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from any_llm.exceptions import ModelNotFoundError
from any_llm.providers.vertexai import VertexaiProvider
from any_llm.providers.vertexai.mistral import (
    convert_mistral_params,
    create_mistral_client,
    is_mistral_model,
    mistral_base_url,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams

BASE_URL = (
    "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/mistralai/models"
)

COMPLETION_BODY = {
    "id": "cmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "mistral-small-2503",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!", "tool_calls": None},
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


def _chunk(content: str, finish_reason: str | None = None) -> str:
    chunk = {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "mistral-small-2503",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


@pytest.fixture
def provider() -> Iterator[VertexaiProvider]:
    with patch("any_llm.providers.vertexai.vertexai.genai.Client") as mock_client:
        api_client = mock_client.return_value._api_client
        api_client.project = "proj"
        api_client.location = "us-central1"
        api_client._async_access_token = AsyncMock(return_value="token-123")
        yield VertexaiProvider()


def _use_transport(provider: VertexaiProvider, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    provider._mistral_client = AsyncOpenAI(
        api_key="unused",
        base_url=BASE_URL,
        http_client=cast("Any", httpx.AsyncClient(transport=httpx.MockTransport(handler))),
        max_retries=0,
    )


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("mistral-small-2503", True),
        ("mistral-medium-3", True),
        ("codestral-2", True),
        ("gemini-2.5-flash", False),
        ("claude-sonnet-4", False),
    ],
)
def test_is_mistral_model(model_id: str, expected: bool) -> None:
    assert is_mistral_model(model_id) is expected


def test_mistral_base_url_regional() -> None:
    assert mistral_base_url("proj", "us-central1") == BASE_URL


def test_mistral_base_url_global() -> None:
    assert (
        mistral_base_url("proj", "global")
        == "https://aiplatform.googleapis.com/v1/projects/proj/locations/global/publishers/mistralai/models"
    )


def test_create_mistral_client_without_timeout_keeps_sdk_default() -> None:
    client = create_mistral_client("proj", "us-central1")
    assert str(client.base_url) == f"{BASE_URL}/"
    assert client.timeout == AsyncOpenAI(api_key="x").timeout


def test_create_mistral_client_with_timeout() -> None:
    assert create_mistral_client("proj", "us-central1", timeout=12.0).timeout == 12.0


def test_convert_mistral_params_basic() -> None:
    params = CompletionParams(
        model_id="mistral-small-2503@001",
        messages=[{"role": "user", "content": "Hi", "extra_content": {"google": {"thought_signature": "x"}}}],
        temperature=0.2,
        max_tokens=50,
        user="someone",
        stream_options={"include_usage": True},
    )

    body = convert_mistral_params(params, safe_prompt=True)

    assert body == {
        "model": "mistral-small-2503",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 50,
        "safe_prompt": True,
    }


def test_convert_mistral_params_maps_max_completion_tokens() -> None:
    params = CompletionParams(
        model_id="mistral-small-2503", messages=[{"role": "user", "content": "Hi"}], max_completion_tokens=20
    )
    body = convert_mistral_params(params)
    assert body["max_tokens"] == 20
    assert "max_completion_tokens" not in body


def test_convert_mistral_params_max_tokens_wins_over_max_completion_tokens() -> None:
    params = CompletionParams(
        model_id="mistral-small-2503",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10,
        max_completion_tokens=20,
    )
    assert convert_mistral_params(params)["max_tokens"] == 10


def test_convert_mistral_params_pydantic_response_format() -> None:
    class Answer(BaseModel):
        value: int

    params = CompletionParams(
        model_id="mistral-small-2503", messages=[{"role": "user", "content": "Hi"}], response_format=Answer
    )
    response_format = convert_mistral_params(params)["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Answer"
    assert response_format["json_schema"]["schema"]["properties"]["value"]["type"] == "integer"


def test_convert_mistral_params_dataclass_response_format() -> None:
    @dataclass
    class Answer:
        value: int

    params = CompletionParams(
        model_id="mistral-small-2503", messages=[{"role": "user", "content": "Hi"}], response_format=Answer
    )
    response_format = convert_mistral_params(params)["response_format"]
    assert response_format["json_schema"]["name"] == "Answer"
    assert "value" in response_format["json_schema"]["schema"]["properties"]


def test_convert_mistral_params_dict_response_format_passes_through() -> None:
    params = CompletionParams(
        model_id="mistral-small-2503",
        messages=[{"role": "user", "content": "Hi"}],
        response_format={"type": "json_object"},
    )
    assert convert_mistral_params(params)["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(("effort", "expected"), [("auto", None), ("none", None), ("low", "high"), ("high", "high")])
def test_convert_mistral_params_reasoning_effort(effort: Any, expected: str | None) -> None:
    params = CompletionParams(
        model_id="mistral-small-2503", messages=[{"role": "user", "content": "Hi"}], reasoning_effort=effort
    )
    assert convert_mistral_params(params).get("reasoning_effort") == expected


@pytest.mark.asyncio
async def test_mistral_completion_calls_raw_predict(provider: VertexaiProvider) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=COMPLETION_BODY)

    _use_transport(provider, handler)

    result = await provider.acompletion(
        model="mistral-small-2503", messages=[{"role": "user", "content": "Hi"}], max_tokens=5
    )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Hello!"
    assert result.usage is not None
    assert result.usage.total_tokens == 5

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == f"{BASE_URL}/mistral-small-2503:rawPredict"
    assert request.headers["Authorization"] == "Bearer token-123"
    assert json.loads(request.content) == {
        "model": "mistral-small-2503",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
        "max_tokens": 5,
    }


@pytest.mark.asyncio
async def test_mistral_completion_streams_from_stream_raw_predict(provider: VertexaiProvider) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        sse = _chunk("Good") + _chunk("bye", finish_reason="stop") + "data: [DONE]\n\n"
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})

    _use_transport(provider, handler)

    stream = await provider.acompletion(model="codestral-2", messages=[{"role": "user", "content": "Hi"}], stream=True)
    chunks = [chunk async for chunk in stream]

    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Goodbye"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert str(requests[0].url) == f"{BASE_URL}/codestral-2:streamRawPredict"
    assert json.loads(requests[0].content)["stream"] is True


@pytest.mark.asyncio
async def test_mistral_completion_forwards_timeout(provider: VertexaiProvider) -> None:
    timeouts: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json=COMPLETION_BODY)

    _use_transport(provider, handler)

    await provider.acompletion(model="mistral-small-2503", messages=[{"role": "user", "content": "Hi"}], timeout=7.0)

    assert timeouts[0]["read"] == 7.0


@pytest.mark.asyncio
async def test_mistral_completion_error_status_is_raised(provider: VertexaiProvider) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": 404, "message": "Publisher model not found."}})

    _use_transport(provider, handler)
    provider._unified_exceptions = True

    with pytest.raises(ModelNotFoundError):
        await provider.acompletion(model="mistral-small-2503", messages=[{"role": "user", "content": "Hi"}])


@pytest.mark.asyncio
async def test_mistral_client_is_created_once_with_vertex_project_and_location(provider: VertexaiProvider) -> None:
    with patch(
        "any_llm.providers.vertexai.vertexai.acompletion_mistral", new=AsyncMock(return_value=MagicMock())
    ) as mock_call:
        await provider._acompletion(
            CompletionParams(model_id="mistral-small-2503", messages=[{"role": "user", "content": "Hi"}])
        )
        first_client = provider._mistral_client
        await provider._acompletion(
            CompletionParams(model_id="mistral-small-2503", messages=[{"role": "user", "content": "Hi"}])
        )

    assert first_client is not None
    assert provider._mistral_client is first_client
    assert str(first_client.base_url) == f"{BASE_URL}/"
    assert mock_call.await_args is not None
    assert mock_call.await_args.args[1] == "token-123"


def test_mistral_client_inherits_provider_timeout() -> None:
    with patch("any_llm.providers.vertexai.vertexai.genai.Client"):
        provider = VertexaiProvider(timeout=30.0)
    assert provider._http_timeout == 30.0


@pytest.mark.asyncio
async def test_gemini_model_does_not_use_mistral_endpoint(provider: VertexaiProvider) -> None:
    with (
        patch("any_llm.providers.gemini.base.GoogleProvider._acompletion", new=AsyncMock()) as google_call,
        patch("any_llm.providers.vertexai.vertexai.acompletion_mistral", new=AsyncMock()) as mistral_call,
    ):
        await provider._acompletion(
            CompletionParams(model_id="gemini-2.5-flash", messages=[{"role": "user", "content": "Hi"}])
        )

    google_call.assert_awaited_once()
    mistral_call.assert_not_awaited()
    assert provider._mistral_client is None
