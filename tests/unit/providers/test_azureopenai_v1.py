import asyncio
import copy
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import BaseModel
from typing_extensions import override

from any_llm.constants import LLMProvider
from any_llm.exceptions import (
    AuthenticationError,
    MissingApiKeyError,
    ProviderError,
    RateLimitError,
    UnsupportedParameterError,
)
from any_llm.providers.azureopenai.azureopenai import AzureopenaiProvider
from any_llm.types.audio import AudioSpeechParams, AudioTranscriptionParams
from any_llm.types.image import ImageGenerationParams
from tests.unit.providers.test_azureopenai_provider import (
    _CHAT_RESPONSE,
    _CHAT_STREAM,
    _RESPONSES_RESPONSE,
    _azure_transport,
)


@pytest.fixture(autouse=True)
def clear_azure_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_VERSION",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_AD_TOKEN",
        "AZURE_OPENAI_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_azure_ci_endpoint_is_fixed(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://unrelated.example")
    config = request.getfixturevalue("provider_client_config")[LLMProvider.AZUREOPENAI]
    assert config["api_base"] == "https://mlrun-me8bof5t-eastus2.cognitiveservices.azure.com/"
    assert config["api_version"] == "v1"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://resource.azure.com",
        "https://resource.azure.com/",
        "https://resource.azure.com/openai/v1",
        "https://resource.azure.com/openai/v1/",
        "https://proxy.example/tenant/openai/v1/",
    ],
)
def test_endpoint_normalization(endpoint: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.example")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    with patch("any_llm.providers.azureopenai.azureopenai.AsyncOpenAI") as client:
        AzureopenaiProvider(api_base=endpoint, api_key="azure-key")
    expected = endpoint.rstrip("/")
    if not expected.endswith("/openai/v1"):
        expected += "/openai/v1"
    assert client.call_args.kwargs["base_url"] == expected + "/"
    assert client.call_args.kwargs["api_key"] == "azure-key"


def test_explicit_v1_overrides_dated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_VERSION", "2025-03-01-preview")
    with patch("any_llm.providers.azureopenai.azureopenai.AsyncOpenAI") as client:
        AzureopenaiProvider(api_key="key", api_base="https://resource.azure.com", api_version="v1")
    assert client.call_args.kwargs["default_query"] is None


@pytest.mark.parametrize(
    ("options", "guidance"),
    [
        ({"azure_deployment": "old"}, "Pass your Azure deployment name as `model`"),
        ({"api_version": "2025-03-01-preview"}, "Remove the dated version"),
    ],
)
def test_configuration_errors_have_migration_guidance(options: dict[str, Any], guidance: str) -> None:
    with pytest.raises(UnsupportedParameterError, match=guidance):
        AzureopenaiProvider(api_key="key", api_base="https://resource.azure.com", **options)


@pytest.mark.parametrize("credential", [{"api_key": ""}, {"azure_ad_token": ""}])
def test_empty_explicit_credentials_never_fall_back(
    credential: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_API_KEY", "generic-key")
    with pytest.raises(MissingApiKeyError):
        AzureopenaiProvider(api_base="https://resource.azure.com", **credential)


@pytest.mark.parametrize("api_key", ["", "key"])
def test_conflicting_explicit_credentials_including_empty(api_key: str) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        AzureopenaiProvider(api_base="https://resource.azure.com", api_key=api_key, azure_ad_token="token")  # noqa: S106


def test_conflicting_explicit_entra_credentials() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        AzureopenaiProvider(
            api_base="https://resource.azure.com",
            azure_ad_token="token",  # noqa: S106
            azure_ad_token_provider=lambda: "other-token",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("credential", [{"api_key": "explicit-key"}, {"azure_ad_token": "explicit-token"}])
async def test_static_auth_options_and_transport_ownership(
    credential: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "ignored-key")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "ignored-token")
    transport, requests = _azure_transport()
    http_client = httpx.AsyncClient(transport=transport)
    provider = AzureopenaiProvider(
        api_base="https://resource.azure.com",
        http_client=http_client,
        **credential,
        default_query={"trace": "yes"},
        default_headers={"x-custom": "preserved"},
        timeout=12,
        max_retries=0,
    )
    try:
        await provider.acompletion(
            model="deployment",
            messages=[{"role": "user", "content": "Hi"}],
            timeout=3,
            extra_headers={"x-request": "yes"},
            extra_query={"request": "yes"},
        )
    finally:
        await provider.client.close()
    request = requests[0]
    assert request.headers["authorization"] == f"Bearer {next(iter(credential.values()))}"
    assert request.headers["x-custom"] == "preserved"
    assert request.headers["x-request"] == "yes"
    assert dict(request.url.params) == {"trace": "yes", "request": "yes"}
    assert request.extensions["timeout"]["read"] == 3
    assert http_client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, "", 123])
async def test_invalid_tokens_send_no_request(result: object) -> None:
    transport, requests = _azure_transport()
    provider = AzureopenaiProvider(
        api_base="https://resource.azure.com",
        azure_ad_token_provider=AsyncMock(return_value=result),
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
    )
    try:
        with (
            pytest.warns(DeprecationWarning, match="Provider-specific exceptions"),
            pytest.raises(ValueError, match="non-empty string"),
        ):
            await provider.acompletion(model="deployment", messages=[{"role": "user", "content": "Hi"}])
        assert requests == []
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_failing_token_provider_sends_no_request() -> None:
    transport, requests = _azure_transport()
    provider = AzureopenaiProvider(
        api_base="https://resource.azure.com",
        azure_ad_token_provider=AsyncMock(side_effect=ValueError("token failed")),
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
    )
    try:
        with (
            pytest.warns(DeprecationWarning, match="Provider-specific exceptions"),
            pytest.raises(ValueError, match="token failed"),
        ):
            await provider.acompletion(model="deployment", messages=[{"role": "user", "content": "Hi"}])
        assert requests == []
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_async_token_provider_refreshes_on_retry() -> None:
    tokens = AsyncMock(side_effect=["first", "second"])
    transport, requests = _azure_transport((429, 200))
    provider = AzureopenaiProvider(
        api_base="https://resource.azure.com",
        azure_ad_token_provider=tokens,
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=1,
    )
    try:
        await provider.acompletion(model="deployment", messages=[{"role": "user", "content": "Hi"}])
        assert [r.headers["authorization"] for r in requests] == ["Bearer first", "Bearer second"]
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_cross_origin_redirect_strips_authorization() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(307, headers={"location": "https://other.example/chat/completions"})
        return httpx.Response(200, json=_CHAT_RESPONSE)

    provider = AzureopenaiProvider(
        api_key="secret-key",
        api_base="https://resource.azure.com",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle), follow_redirects=True),
        max_retries=0,
    )
    try:
        await provider.acompletion(model="deployment", messages=[{"role": "user", "content": "Hi"}])
        assert requests[0].headers["authorization"] == "Bearer secret-key"
        assert "authorization" not in requests[1].headers
    finally:
        await provider.client.close()


class Answer(BaseModel):
    answer: str


@dataclass
class DataclassAnswer:
    answer: str


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_format",
    [
        Answer,
        DataclassAnswer,
        {
            "type": "json_schema",
            "name": "Answer",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ],
)
async def test_structured_responses_wire_schema_and_output(response_format: Any) -> None:
    requests: list[httpx.Request] = []
    body: dict[str, Any] = copy.deepcopy(_RESPONSES_RESPONSE)
    body["output"][0]["content"][0]["text"] = '{"answer":"yes"}'

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=body)

    provider = AzureopenaiProvider(
        api_key="key",
        api_base="https://resource.azure.com",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    try:
        response = await provider.aresponses(model="deployment", input_data="Answer", response_format=response_format)
        if response_format in (Answer, DataclassAnswer):
            assert response.output_parsed.answer == "yes"
        else:
            assert response.output[0].content[0].text == '{"answer":"yes"}'
        schema = json.loads(requests[0].content)["text"]["format"]
        assert schema["type"] == "json_schema"
        assert schema["schema"]["properties"]["answer"]["type"] == "string"
        assert requests[0].url.path == "/openai/v1/responses"
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_models_and_embeddings_use_v1() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "deployment", "object": "model", "created": 1, "owned_by": "azure"},
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "embedding-deployment",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    provider = AzureopenaiProvider(
        api_key="key",
        api_base="https://resource.azure.com",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    try:
        assert (await provider.alist_models())[0].id == "deployment"
        result = await provider.aembedding(model="embedding-deployment", inputs="Hi", dimensions=2)
        assert result.data[0].embedding == [0.1, 0.2]
        assert [r.url.path for r in requests] == ["/openai/v1/models", "/openai/v1/embeddings"]
        assert json.loads(requests[1].content)["dimensions"] == 2
    finally:
        await provider.client.close()


def test_sync_chat_responses_and_streaming() -> None:
    transport, requests = _azure_transport()
    provider = AzureopenaiProvider(
        api_key="key",
        api_base="https://resource.azure.com",
        http_client=httpx.AsyncClient(transport=transport),
    )
    try:
        messages: list[Any] = [{"role": "user", "content": "Hi"}]
        assert (
            provider.completion(model="deployment", messages=messages).choices[0].message.content == "Hello from Azure"
        )
        assert provider.responses(model="deployment", input_data="Hello").id == "resp-test"
        assert len(list(provider.completion(model="deployment", messages=messages, stream=True))) == 1
        assert len(requests) == 3
    finally:
        asyncio.run(provider.client.close())


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, "preview", "v1"])
async def test_media_uses_scoped_v1_preview_and_preserves_options(version: str | None) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/images/generations"):
            return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "aW1hZ2U="}]})
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "Hello"})
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(200, content=b"audio", headers={"content-type": "audio/mpeg"})
        return httpx.Response(200, json=_CHAT_RESPONSE)

    provider = AzureopenaiProvider(
        api_key="key",
        api_base="https://resource.azure.com",
        default_query={"trace": "yes", "api-version": "v1"},
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    query = {"request": "yes"}
    if version:
        query["api-version"] = version
    try:
        if version is None:
            image = await provider.aimage_generation(model="image-deployment", prompt="A cat")
            transcription = await provider.atranscription(model="audio-deployment", file=b"audio")
            speech = await provider.aspeech(model="speech-deployment", input="Hello", voice="alloy")
        else:
            image = await provider._aimage_generation(
                ImageGenerationParams(model_id="image-deployment", prompt="A cat"),
                extra_query=query,
            )
            transcription = await provider._atranscription(
                AudioTranscriptionParams(model_id="audio-deployment", file=b"audio"),
                extra_query=query,
            )
            speech = await provider._aspeech(
                AudioSpeechParams(model_id="speech-deployment", input="Hello", voice="alloy"),
                extra_query=query,
            )
        await provider.acompletion(model="deployment", messages=[{"role": "user", "content": "Hi"}])
        assert image.data is not None
        assert image.data[0].b64_json == "aW1hZ2U="
        assert transcription.text == "Hello"
        assert speech == b"audio"
        assert [r.url.path for r in requests[:3]] == [
            "/openai/v1/images/generations",
            "/openai/v1/audio/transcriptions",
            "/openai/v1/audio/speech",
        ]
        for request in requests[:3]:
            expected_query = {"trace": "yes", "api-version": version or "preview"}
            if version is not None:
                expected_query["request"] = "yes"
            assert dict(request.url.params) == expected_query
            assert request.headers["authorization"] == "Bearer key"
        assert requests[3].url.params["api-version"] == "v1"
        assert json.loads(requests[0].content)["model"] == "image-deployment"
        assert b"audio-deployment" in requests[1].content
        assert json.loads(requests[2].content)["voice"] == "alloy"
        assert query == ({"request": "yes", "api-version": version} if version else {"request": "yes"})
    finally:
        await provider.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "error"), [(401, AuthenticationError), (429, RateLimitError)])
async def test_v1_http_errors_map_to_unified_exceptions(
    status: int,
    error: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    transport, requests = _azure_transport((status,))
    provider = AzureopenaiProvider(
        api_key="key",
        api_base="https://resource.azure.com",
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
    )
    try:
        with pytest.raises(error):
            await provider.acompletion(model="deployment", messages=[{"role": "user", "content": "Hi"}])
        assert len(requests) == 1
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_v1_timeout_maps_to_unified_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

    def handle(request: httpx.Request) -> httpx.Response:
        message = "read timed out"
        raise httpx.ReadTimeout(message, request=request)

    provider = AzureopenaiProvider(
        api_key="key",
        api_base="https://resource.azure.com",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        max_retries=0,
    )
    try:
        with pytest.raises(ProviderError):
            await provider.aresponses(model="deployment", input_data="Hi", timeout=1)
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_v1_chat_preserves_response_fields() -> None:
    transport, _ = _azure_transport()
    provider = AzureopenaiProvider(
        api_key="key",
        api_base="https://resource.azure.com",
        http_client=httpx.AsyncClient(transport=transport),
    )
    try:
        response = await provider.acompletion(model="deployment", messages=[{"role": "user", "content": "Hi"}])
        body = response.model_dump()
        assert body["future_response_field"] is True
        assert body["choices"][0]["message"]["future_message_field"] is True
        assert response.usage is not None
        assert response.usage.total_tokens == 3
    finally:
        await provider.client.close()


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self, mode: str, api: str) -> None:
        self.closed = False
        self.mode = mode
        self.api = api
        self.waiting = asyncio.Event()

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self.api == "chat":
            yield _CHAT_STREAM.split(b"data: [DONE]")[0]
        else:
            yield (
                b'data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg",'
                b'"output_index":0,"content_index":0,"delta":"Hi"}\n\n'
            )
        if self.mode == "failure":
            message = "stream failed"
            raise ValueError(message)
        if self.mode == "cancellation":
            self.waiting.set()
            await asyncio.Event().wait()
        yield b"data: [DONE]\n\n"

    @override
    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["exhaustion", "early_exit", "failure", "cancellation"])
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_stream_releases_transport(mode: str, api: str) -> None:
    body = TrackedStream(mode, api)
    provider = AzureopenaiProvider(
        api_key="key",
        api_base="https://resource.azure.com",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=body,
                )
            )
        ),
    )
    try:
        stream = (
            await provider.acompletion(model="deployment", messages=[{"role": "user", "content": "Hi"}], stream=True)
            if api == "chat"
            else await provider.aresponses(model="deployment", input_data="Hi", stream=True)
        )
        assert isinstance(stream, AsyncIterator)
        await anext(stream)
        if mode == "early_exit":
            await stream.aclose()  # type: ignore[union-attr]
        elif mode == "failure":
            with (
                pytest.warns(DeprecationWarning, match="Provider-specific exceptions"),
                pytest.raises(ValueError, match="stream failed"),
            ):
                await anext(stream)
        elif mode == "cancellation":
            task = asyncio.ensure_future(anext(stream))
            await body.waiting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif mode == "exhaustion":
            assert [chunk async for chunk in stream] == []
        assert body.closed
    finally:
        await provider.client.close()
