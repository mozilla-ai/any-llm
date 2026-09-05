import json
import re
import threading
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import OpenAIError

from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.azureopenai.azureopenai import AzureopenaiProvider

_HTTP_OK = 200
_HTTP_TOO_MANY_REQUESTS = 429
_CHAT_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "deployment-name",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "logprobs": None,
            "message": {"role": "assistant", "content": "Hello from Azure", "future_message_field": True},
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    "future_response_field": True,
}
_CHAT_STREAM = (
    b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,'
    b'"model":"deployment-name","choices":[{"index":0,"delta":{"role":"assistant",'
    b'"content":"Hello"},"finish_reason":null}]}\n\n'
    b"data: [DONE]\n\n"
)
_RESPONSES_RESPONSE = {
    "id": "resp-test",
    "object": "response",
    "created_at": 1,
    "model": "deployment-name",
    "output": [
        {
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Hello from Responses", "annotations": []}],
        }
    ],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
    "future_response_field": True,
}


def _azure_transport(
    status_codes: tuple[int, ...] = (_HTTP_OK,),
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        request.read()
        requests.append(request)
        status = status_codes[min(len(requests) - 1, len(status_codes) - 1)]
        if status != _HTTP_OK:
            return httpx.Response(
                status,
                headers={"Retry-After": "0"},
                json={
                    "error": {
                        "message": "request rejected",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
            )
        if request.url.path.endswith("/responses"):
            return httpx.Response(_HTTP_OK, json=_RESPONSES_RESPONSE)
        if b'"stream":true' in request.content:
            return httpx.Response(_HTTP_OK, headers={"Content-Type": "text/event-stream"}, content=_CHAT_STREAM)
        return httpx.Response(_HTTP_OK, json=_CHAT_RESPONSE)

    return httpx.MockTransport(handle), requests


@pytest.mark.parametrize("api_version", [None, "v1"])
@pytest.mark.parametrize("environment_version", ["", "v1"])
def test_azureopenai_normalizes_v1_endpoint_and_preserves_client_options(
    api_version: str | None, environment_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_VERSION", environment_version)
    with patch("any_llm.providers.azureopenai.azureopenai.AsyncOpenAI") as sdk_client:
        AzureopenaiProvider(
            api_key="key",
            api_base="https://explicit.openai.azure.com/openai/v1/",
            azure_endpoint="https://ignored.openai.azure.com",
            api_version=api_version,
            default_query={"api-version": "v1", "trace": "1"},
            timeout=30,
        )

    sdk_client.assert_called_once_with(
        api_key="key",
        base_url="https://explicit.openai.azure.com/openai/v1/",
        default_query={"api-version": "v1", "trace": "1"},
        timeout=30,
    )


def test_azureopenai_uses_environment_endpoint_and_entra_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://resource.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "entra-token")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "lower-precedence-key")

    with patch("any_llm.providers.azureopenai.azureopenai.AsyncOpenAI") as sdk_client:
        AzureopenaiProvider()

    assert sdk_client.call_args.kwargs["api_key"] == "entra-token"
    assert sdk_client.call_args.kwargs["base_url"] == "https://resource.openai.azure.com/openai/v1/"


def test_azureopenai_explicit_azure_arguments_override_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://environment.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "environment-key")

    with patch("any_llm.providers.azureopenai.azureopenai.AsyncOpenAI") as sdk_client:
        AzureopenaiProvider(
            azure_endpoint="https://explicit.openai.azure.com",
            azure_ad_token="explicit-token",  # noqa: S106
        )

    assert sdk_client.call_args.kwargs["api_key"] == "explicit-token"
    assert sdk_client.call_args.kwargs["base_url"] == "https://explicit.openai.azure.com/openai/v1/"


@pytest.mark.parametrize(
    ("legacy_options", "expected_parameter"),
    [
        ({"api_version": "2024-10-21"}, "api_version"),
        ({"azure_deployment": "deployment-name"}, "azure_deployment"),
        ({"default_query": {"api-version": "2024-10-21"}}, "default_query['api-version']"),
    ],
)
def test_azureopenai_rejects_legacy_routing_options(legacy_options: dict[str, object], expected_parameter: str) -> None:
    with pytest.raises(UnsupportedParameterError, match=re.escape(expected_parameter)):
        AzureopenaiProvider(
            api_key="key",
            api_base="https://resource.openai.azure.com",
            **legacy_options,
        )


def test_azureopenai_rejects_legacy_api_version_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-10-21")

    with pytest.raises(UnsupportedParameterError, match="OPENAI_API_VERSION"):
        AzureopenaiProvider(api_key="key", api_base="https://resource.openai.azure.com")


@pytest.mark.parametrize(
    ("api_key", "ad_token", "expected_credential"),
    [("", "entra-token", "entra-token"), ("api-key", "", "api-key")],
)
def test_azureopenai_treats_empty_environment_credentials_as_absent(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str,
    ad_token: str,
    expected_credential: str,
) -> None:
    """An empty environment value must not count as a second credential."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://resource.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", api_key)
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", ad_token)

    with patch("any_llm.providers.azureopenai.azureopenai.AsyncOpenAI") as sdk_client:
        AzureopenaiProvider()

    assert sdk_client.call_args.kwargs["api_key"] == expected_credential


def test_azureopenai_requires_one_endpoint_and_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_AD_TOKEN", "AZURE_OPENAI_API_KEY"):
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(ValueError, match="endpoint is required"):
        AzureopenaiProvider(api_key="key")
    with pytest.raises(MissingApiKeyError, match="AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN"):
        AzureopenaiProvider(api_base="https://resource.openai.azure.com")
    token = "token"  # noqa: S105
    with pytest.raises(OpenAIError, match="mutually exclusive"):
        AzureopenaiProvider(
            api_key="key",
            api_base="https://resource.openai.azure.com",
            azure_ad_token=token,
        )
    with pytest.raises(OpenAIError, match="mutually exclusive"):
        AzureopenaiProvider(
            api_key="key",
            api_base="https://resource.openai.azure.com",
            azure_ad_token_provider=lambda: token,
        )


@pytest.mark.asyncio
async def test_azureopenai_sends_chat_to_v1_with_bearer_api_key_and_preserves_optional_presence() -> None:
    transport, requests = _azure_transport()
    provider = AzureopenaiProvider(
        api_key="api-key",
        api_base="https://resource.openai.azure.com",
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
    )
    try:
        first = await provider.acompletion(
            model="deployment-name",
            messages=[{"role": "user", "content": "Hello"}],
        )
        await provider.acompletion(
            model="deployment-name",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0,
        )
    finally:
        await provider.client.close()

    assert first.choices[0].message.content == "Hello from Azure"
    assert [request.url.path for request in requests] == [
        "/openai/v1/chat/completions",
        "/openai/v1/chat/completions",
    ]
    assert [request.headers["Authorization"] for request in requests] == ["Bearer api-key", "Bearer api-key"]
    first_body = json.loads(requests[0].content)
    second_body = json.loads(requests[1].content)
    assert first_body["model"] == "deployment-name"
    assert "temperature" not in first_body
    assert second_body["temperature"] == 0


@pytest.mark.asyncio
async def test_azureopenai_refreshes_sync_entra_provider_before_sdk_retry() -> None:
    """The SDK retries with refreshed credentials without blocking the event loop."""
    tokens = iter(("entra-token-1", "entra-token-2"))
    event_loop_thread = threading.get_ident()
    provider_threads: list[int] = []

    def token_provider() -> str:
        provider_threads.append(threading.get_ident())
        return next(tokens)

    transport, requests = _azure_transport((_HTTP_TOO_MANY_REQUESTS, _HTTP_OK))
    provider = AzureopenaiProvider(
        api_base="https://resource.openai.azure.com",
        azure_ad_token_provider=token_provider,
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=1,
    )
    try:
        result = await provider.acompletion(
            model="deployment-name",
            messages=[{"role": "user", "content": "Hello"}],
        )
    finally:
        await provider.client.close()

    assert result.choices[0].message.content == "Hello from Azure"
    assert [request.url.path for request in requests] == [
        "/openai/v1/chat/completions",
        "/openai/v1/chat/completions",
    ]
    assert [request.headers["Authorization"] for request in requests] == [
        "Bearer entra-token-1",
        "Bearer entra-token-2",
    ]
    assert provider_threads
    assert all(thread_id != event_loop_thread for thread_id in provider_threads)


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["", 1])
async def test_azureopenai_rejects_invalid_entra_provider_token(token: object) -> None:
    """Reject empty and non-string dynamic tokens before sending the request."""
    transport, _ = _azure_transport()
    provider = AzureopenaiProvider(
        api_base="https://resource.openai.azure.com",
        azure_ad_token_provider=AsyncMock(return_value=token),
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
    )
    try:
        with (
            pytest.warns(DeprecationWarning, match="Provider-specific exceptions"),
            pytest.raises(ValueError, match="non-empty string"),
        ):
            await provider.acompletion(
                model="deployment-name",
                messages=[{"role": "user", "content": "Hello"}],
            )
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_azureopenai_supports_async_entra_provider_and_chat_sse() -> None:
    async def token_provider() -> str:
        return "async-entra-token"

    transport, requests = _azure_transport()
    provider = AzureopenaiProvider(
        api_base="https://resource.openai.azure.com",
        azure_ad_token_provider=token_provider,
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
    )
    try:
        stream = await provider.acompletion(
            model="deployment-name",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]
    finally:
        await provider.client.close()

    assert [request.url.path for request in requests] == ["/openai/v1/chat/completions"]
    assert [request.headers["Authorization"] for request in requests] == ["Bearer async-entra-token"]
    assert [chunk.choices[0].delta.content for chunk in chunks] == ["Hello"]


@pytest.mark.asyncio
async def test_azureopenai_sends_responses_to_v1() -> None:
    transport, requests = _azure_transport()
    provider = AzureopenaiProvider(
        api_key="api-key",
        api_base="https://resource.openai.azure.com",
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
    )
    try:
        response = await provider.aresponses(model="deployment-name", input_data="Hello")
    finally:
        await provider.client.close()

    assert response.id == "resp-test"
    assert [request.url.path for request in requests] == ["/openai/v1/responses"]
    assert [request.headers["Authorization"] for request in requests] == ["Bearer api-key"]
    assert requests[0].read() == b'{"input":"Hello","model":"deployment-name"}'
