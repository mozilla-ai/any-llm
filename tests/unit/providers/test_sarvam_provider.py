import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from sarvamai import CreateChatCompletionV2Response, ModelListResponse
from sarvamai.types.chat_completion_chunk import ChatCompletionChunk as SarvamChatCompletionChunk

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import AuthenticationError, MissingApiKeyError, RateLimitError
from any_llm.providers.sarvam import SarvamProvider
from any_llm.providers.sarvam.utils import _convert_chunk, _convert_models_list, _convert_response
from any_llm.types.completion import ChatCompletion, CompletionParams

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "Hi"}]

COMPLETION_BODY: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "sarvam-105b",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Namaste!", "reasoning_content": "Greet back."},
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12,
        "completion_tokens_details": {"reasoning_tokens": 3},
        "prompt_tokens_details": {"cached_tokens": 2},
    },
}

STREAM_EVENTS: list[dict[str, Any]] = [
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "sarvam-105b",
        "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "Hmm"}}],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "sarvam-105b",
        "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": "stop"}],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "sarvam-105b",
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    },
]


class _Recorder:
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)

    @property
    def last_body(self) -> dict[str, Any]:
        body: dict[str, Any] = json.loads(self.requests[-1].content)
        return body


def _provider(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> tuple[SarvamProvider, _Recorder]:
    recorder = _Recorder(handler)
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return SarvamProvider(api_key="test-key", httpx_client=client, **kwargs), recorder


def _sse(events: list[dict[str, Any]]) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def test_provider_metadata() -> None:
    provider = SarvamProvider(api_key="test-key")
    assert provider.PROVIDER_NAME == "sarvam"
    assert provider.ENV_API_KEY_NAME == "SARVAM_API_KEY"
    assert provider.ENV_API_BASE_NAME == "SARVAM_API_BASE"
    assert provider.PROVIDER_DOCUMENTATION_URL == "https://docs.sarvam.ai/"
    assert LLMProvider("sarvam") is LLMProvider.SARVAM
    assert AnyLLM.get_provider_class("sarvam") is SarvamProvider


def test_capability_flags() -> None:
    assert SarvamProvider.SUPPORTS_COMPLETION
    assert SarvamProvider.SUPPORTS_COMPLETION_STREAMING
    assert SarvamProvider.SUPPORTS_COMPLETION_REASONING
    assert SarvamProvider.SUPPORTS_LIST_MODELS
    assert not SarvamProvider.SUPPORTS_COMPLETION_IMAGE
    assert not SarvamProvider.SUPPORTS_COMPLETION_PDF
    assert not SarvamProvider.SUPPORTS_EMBEDDING
    assert not SarvamProvider.SUPPORTS_RESPONSES
    assert not SarvamProvider.SUPPORTS_BATCH
    assert not SarvamProvider.SUPPORTS_RERANK


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        SarvamProvider()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SARVAM_API_KEY", "env-key")
    provider = SarvamProvider()
    assert provider.client._client_wrapper.api_subscription_key == "env-key"


def test_default_base_is_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SARVAM_API_BASE", raising=False)
    provider = SarvamProvider(api_key="test-key")
    assert provider.client._client_wrapper.get_environment().base == "https://api.sarvam.ai"


def test_api_base_from_env_replaces_only_the_rest_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SARVAM_API_BASE", "https://proxy.internal/")
    provider = SarvamProvider(api_key="test-key")
    environment = provider.client._client_wrapper.get_environment()
    assert environment.base == "https://proxy.internal"
    assert environment.production == "wss://api.sarvam.ai"


@pytest.mark.asyncio
async def test_completion_sends_v2_request_and_converts_response() -> None:
    provider, recorder = _provider(lambda _: httpx.Response(200, json=COMPLETION_BODY))

    result = await provider.acompletion(
        model="sarvam-105b",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "extra_content": {"deepseek": {"x": 1}}},
            {"role": "user", "content": "Again"},
        ],
        temperature=0.2,
        reasoning_effort="low",
        user="user-1",
        timeout=12,
    )

    request = recorder.requests[-1]
    assert str(request.url) == "https://api.sarvam.ai/v2/chat/completions"
    assert request.headers["api-subscription-key"] == "test-key"
    assert request.extensions["timeout"]["read"] == 12
    body = recorder.last_body
    assert body["model"] == "sarvam-105b"
    assert body["temperature"] == 0.2
    assert body["reasoning_effort"] == "low"
    assert body["user"] == "user-1"
    assert "extra_content" not in body["messages"][1]

    assert isinstance(result, ChatCompletion)
    message = result.choices[0].message
    assert message.content == "Namaste!"
    assert message.reasoning is not None
    assert message.reasoning.content == "Greet back."
    assert result.usage is not None
    assert result.usage.total_tokens == 12
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 2
    assert result.usage.completion_tokens_details is not None
    assert result.usage.completion_tokens_details.reasoning_tokens == 3


@pytest.mark.asyncio
async def test_completion_streams_chunks_including_usage_chunk() -> None:
    provider, recorder = _provider(
        lambda _: httpx.Response(200, content=_sse(STREAM_EVENTS), headers={"content-type": "text/event-stream"})
    )

    stream = await provider.acompletion(
        model="sarvam-105b",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = [chunk async for chunk in stream]

    assert recorder.last_body["stream"] is True
    assert recorder.last_body["stream_options"] == {"include_usage": True}
    assert len(chunks) == 3
    assert chunks[0].choices[0].delta.reasoning is not None
    assert chunks[0].choices[0].delta.reasoning.content == "Hmm"
    assert chunks[1].choices[0].delta.content == "Hi"
    assert chunks[1].choices[0].finish_reason == "stop"
    assert chunks[2].choices == []
    assert chunks[2].usage is not None
    assert chunks[2].usage.total_tokens == 3


@pytest.mark.asyncio
async def test_structured_output_sends_json_schema_and_parses() -> None:
    class Answer(BaseModel):
        city: str

    body = {
        **COMPLETION_BODY,
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"city": "Pune"}'}}
        ],
    }
    provider, recorder = _provider(lambda _: httpx.Response(200, json=body))

    result = await provider.acompletion(
        model="sarvam-105b", messages=[{"role": "user", "content": "City?"}], response_format=Answer
    )

    response_format = recorder.last_body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Answer"
    assert response_format["json_schema"]["schema"]["properties"]["city"]["type"] == "string"
    assert result.choices[0].message.parsed == Answer(city="Pune")


@pytest.mark.asyncio
async def test_max_retries_is_sent_with_every_request() -> None:
    calls = {"count": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after": "0"})

    provider, _ = _provider(handler, max_retries=1, unified_exceptions=True)

    with pytest.raises(RateLimitError):
        await provider.acompletion(model="sarvam-105b", messages=[{"role": "user", "content": "Hi"}])

    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_forbidden_maps_to_authentication_error() -> None:
    provider, _ = _provider(
        lambda _: httpx.Response(403, json={"error": {"message": "bad key", "code": "invalid_api_key_error"}}),
        unified_exceptions=True,
    )

    with pytest.raises(AuthenticationError):
        await provider.acompletion(model="sarvam-105b", messages=[{"role": "user", "content": "Hi"}])


@pytest.mark.asyncio
async def test_list_models() -> None:
    models_body = {
        "object": "list",
        "data": [{"id": "sarvam-105b", "object": "model", "created": 1, "owned_by": "sarvam", "access": "standard"}],
    }
    provider, recorder = _provider(lambda _: httpx.Response(200, json=models_body), max_retries=0)

    models = await provider.alist_models()

    assert str(recorder.requests[-1].url) == "https://api.sarvam.ai/v2/models"
    assert [(m.id, m.owned_by, m.created) for m in models] == [("sarvam-105b", "sarvam", 1)]


def test_convert_params_drops_auto_reasoning_effort() -> None:
    params = CompletionParams(model_id="m", messages=MESSAGES, reasoning_effort="auto")
    assert "reasoning_effort" not in SarvamProvider._convert_completion_params(params)


def test_convert_params_maps_max_completion_tokens_to_max_tokens() -> None:
    params = CompletionParams(model_id="m", messages=MESSAGES, max_completion_tokens=50)
    assert SarvamProvider._convert_completion_params(params)["max_tokens"] == 50


def test_convert_params_max_tokens_wins_over_max_completion_tokens() -> None:
    params = CompletionParams(model_id="m", messages=MESSAGES, max_tokens=10, max_completion_tokens=50)
    assert SarvamProvider._convert_completion_params(params)["max_tokens"] == 10


def test_convert_params_drops_stream_options_when_not_streaming() -> None:
    params = CompletionParams(model_id="m", messages=MESSAGES, stream_options={"include_usage": True})
    assert "stream_options" not in SarvamProvider._convert_completion_params(params)


def test_convert_params_passes_dict_response_format_through() -> None:
    params = CompletionParams(model_id="m", messages=MESSAGES, response_format={"type": "json_object"})
    assert SarvamProvider._convert_completion_params(params)["response_format"] == {"type": "json_object"}


def test_convert_params_merges_caller_request_options() -> None:
    params = CompletionParams(model_id="m", messages=MESSAGES, user="u")
    converted = SarvamProvider._convert_completion_params(
        params,
        timeout=5,
        request_options={"timeout_in_seconds": 9, "additional_body_parameters": {"user": "caller"}},
    )
    assert converted["request_options"] == {
        "timeout_in_seconds": 9,
        "additional_body_parameters": {"user": "caller"},
    }
    assert "user" not in converted


def test_convert_params_without_extras_sends_no_request_options() -> None:
    params = CompletionParams(model_id="m", messages=MESSAGES, temperature=0.5)
    assert SarvamProvider._convert_completion_params(params) == {"temperature": 0.5}


def test_request_options_without_max_retries_is_none() -> None:
    provider = SarvamProvider(api_key="test-key")
    assert provider._request_options(None) is None
    assert provider._request_options({"max_retries": 4}) == {"max_retries": 4}


def test_request_options_caller_max_retries_wins() -> None:
    provider = SarvamProvider(api_key="test-key", max_retries=1)
    assert provider._request_options(None) == {"max_retries": 1}
    assert provider._request_options({"max_retries": 4}) == {"max_retries": 4}


def test_convert_response_with_tool_calls() -> None:
    response = CreateChatCompletionV2Response.model_validate(
        {
            **COMPLETION_BODY,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Pune"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": None,
        }
    )

    result = _convert_response(response)

    tool_call = result.choices[0].message.tool_calls[0]  # type: ignore[index]
    assert tool_call.function.name == "get_weather"  # type: ignore[union-attr]
    assert tool_call.function.arguments == '{"city": "Pune"}'  # type: ignore[union-attr]
    assert result.choices[0].message.reasoning is None
    assert result.usage is None


def test_convert_chunk_without_reasoning() -> None:
    chunk = SarvamChatCompletionChunk.model_validate(STREAM_EVENTS[1])
    result = _convert_chunk(chunk)
    assert result.choices[0].delta.content == "Hi"
    assert result.choices[0].delta.reasoning is None


def test_convert_models_list() -> None:
    response = ModelListResponse.model_validate(
        {"data": [{"id": "glm5.3", "created": 2, "owned_by": "sarvam", "access": "provisioned"}]}
    )
    assert [m.id for m in _convert_models_list(response)] == ["glm5.3"]


def test_embedding_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        SarvamProvider._convert_embedding_params(None)
    with pytest.raises(NotImplementedError):
        SarvamProvider._convert_embedding_response(None)
