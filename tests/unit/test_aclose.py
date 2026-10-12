import asyncio
import gc
import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from typing_extensions import override

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.providers.openai.base import BaseOpenAIProvider

# bedrock and sagemaker hold synchronous boto3 clients, which are not bound to an event loop;
# lmstudio, voyage and watsonx open a client per request. Every other provider keeps an async
# client between calls and must close it.
_NO_ASYNC_CLIENT_HELD = {
    LLMProvider.BEDROCK,
    LLMProvider.LMSTUDIO,
    LLMProvider.SAGEMAKER,
    LLMProvider.VOYAGE,
    LLMProvider.WATSONX,
}


def _create_kwargs(provider: LLMProvider) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": "test_key", "api_base": "https://test.example.com"}
    if provider == LLMProvider.BEDROCK:
        kwargs["region_name"] = "us-east-1"
    if provider == LLMProvider.VERTEXAI:
        kwargs["project"] = "test-project"
        kwargs["location"] = "test-location"
    return kwargs


def test_every_provider_that_holds_an_async_client_overrides_aclose(provider: LLMProvider) -> None:
    provider_class = AnyLLM.get_provider_class(provider)

    overrides_aclose = provider_class.aclose is not AnyLLM.aclose

    assert overrides_aclose is (provider not in _NO_ASYNC_CLIENT_HELD)


@pytest.mark.asyncio
async def test_aclose_can_be_called_twice_on_every_provider(provider: LLMProvider) -> None:
    if provider == LLMProvider.SAGEMAKER:
        pytest.skip("sagemaker requires AWS credentials on instantiation")
    if sys.version_info >= (3, 14) and provider.value in ("voyage", "watsonx"):
        pytest.skip(f"{provider.value} is not compatible with Python 3.14+")
    llm = AnyLLM.create(provider.value, **_create_kwargs(provider))

    await llm.aclose()
    await llm.aclose()


@pytest.mark.asyncio
async def test_async_with_returns_the_provider_and_closes_it_on_exit() -> None:
    async with AnyLLM.create("openai", api_key="test_key") as llm:
        assert isinstance(llm, BaseOpenAIProvider)
        assert not llm.client.is_closed()

    assert llm.client.is_closed()


@pytest.mark.asyncio
async def test_async_with_closes_the_provider_when_the_body_raises() -> None:
    llm = AnyLLM.create("openai", api_key="test_key")
    assert isinstance(llm, BaseOpenAIProvider)

    async def fail_inside() -> None:
        async with llm:
            msg = "boom"
            raise ValueError(msg)

    with pytest.raises(ValueError, match="boom"):
        await fail_inside()

    assert llm.client.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "deepseek", "anthropic", "groq", "cerebras", "together"])
async def test_aclose_closes_the_sdk_client(provider: str) -> None:
    llm = AnyLLM.create(provider, api_key="test_key")

    await llm.aclose()

    assert llm.client.is_closed()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_aclose_closes_an_openai_compatible_endpoint_client() -> None:
    llm = AnyLLM.create_openai_compatible("mygateway", api_base="https://gw.example/v1")
    assert isinstance(llm, BaseOpenAIProvider)

    await llm.aclose()

    assert llm.client.is_closed()


@pytest.mark.asyncio
async def test_meta_closes_both_of_its_clients() -> None:
    from any_llm.providers.meta.meta import MetaProvider

    llm = AnyLLM.create("meta", api_key="test_key")
    assert isinstance(llm, MetaProvider)

    await llm.aclose()

    assert llm.client.is_closed()
    assert llm._anthropic_client.is_closed()


@pytest.mark.asyncio
async def test_otari_closes_its_gateway_client() -> None:
    from any_llm.providers.otari.otari import OtariProvider

    llm = AnyLLM.create("otari", api_key="test_key", api_base="https://gateway.example")
    assert isinstance(llm, OtariProvider)

    await llm.aclose()

    assert llm.otari_client._http.is_closed


@pytest.mark.asyncio
async def test_gemini_closes_its_async_http_client() -> None:
    from any_llm.providers.gemini.base import GoogleProvider

    llm = AnyLLM.create("gemini", api_key="test_key")
    assert isinstance(llm, GoogleProvider)

    await llm.aclose()

    assert llm.client._api_client._async_httpx_client.is_closed


@pytest.mark.asyncio
async def test_vertexai_closes_its_partner_and_mistral_clients() -> None:
    from any_llm.providers.vertexai import VertexaiProvider
    from any_llm.providers.vertexai.mistral import create_mistral_client

    llm = AnyLLM.create("vertexai", project="test-project", location="us-central1")
    assert isinstance(llm, VertexaiProvider)
    partner_provider = llm._get_partner_provider()
    llm._mistral_client = create_mistral_client(project="test-project", location="us-central1")

    await llm.aclose()

    assert llm.client._api_client._async_httpx_client.is_closed
    assert partner_provider.client.is_closed()
    assert llm._mistral_client.is_closed()


@pytest.mark.asyncio
async def test_vertexai_leaves_a_caller_supplied_httpx_client_open() -> None:
    """genai never closes an ``httpx_async_client`` passed in ``http_options``; the partner client shares it."""
    from any_llm.providers.vertexai import VertexaiProvider

    async with httpx.AsyncClient() as caller_client:
        llm = AnyLLM.create(
            "vertexai",
            project="test-project",
            location="us-central1",
            http_options={"httpx_async_client": caller_client},
        )
        assert isinstance(llm, VertexaiProvider)
        llm._get_partner_provider()

        await llm.aclose()

        assert not caller_client.is_closed


@pytest.mark.asyncio
async def test_mistral_closes_the_async_client_it_built() -> None:
    from any_llm.providers.mistral import MistralProvider

    llm = AnyLLM.create("mistral", api_key="test_key")
    assert isinstance(llm, MistralProvider)
    async_client = llm.client.sdk_configuration.async_client
    assert async_client is not None

    await llm.aclose()

    assert async_client.is_closed  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mistral_leaves_a_caller_supplied_async_client_open() -> None:
    async with httpx.AsyncClient() as caller_client:
        llm = AnyLLM.create("mistral", api_key="test_key", async_client=caller_client)

        await llm.aclose()

        assert not caller_client.is_closed


@pytest.mark.asyncio
async def test_cohere_closes_its_http_client() -> None:
    from any_llm.providers.cohere import CohereProvider

    llm = AnyLLM.create("cohere", api_key="test_key")
    assert isinstance(llm, CohereProvider)

    await llm.aclose()

    assert llm.client._client_wrapper.httpx_client.httpx_client.is_closed


@pytest.mark.asyncio
async def test_ollama_closes_its_http_client() -> None:
    from any_llm.providers.ollama import OllamaProvider

    llm = AnyLLM.create("ollama")
    assert isinstance(llm, OllamaProvider)

    await llm.aclose()

    assert llm.client._client.is_closed


@pytest.mark.asyncio
async def test_huggingface_closes_both_of_its_clients() -> None:
    from any_llm.providers.huggingface import HuggingfaceProvider

    llm = AnyLLM.create("huggingface", api_key="test_key")
    assert isinstance(llm, HuggingfaceProvider)
    # The inference client opens its HTTP session on first use.
    http_client = await llm.client._get_async_client()  # type: ignore[no-untyped-call]

    await llm.aclose()

    assert http_client.is_closed
    assert llm.responses_client.is_closed()


@pytest.mark.asyncio
async def test_azure_closes_its_chat_and_embeddings_clients() -> None:
    from any_llm.providers.azure import AzureProvider

    llm = AnyLLM.create("azure", api_key="test_key", api_base="https://test.example.com")
    assert isinstance(llm, AzureProvider)

    with (
        patch.object(llm.chat_client, "close", new=AsyncMock()) as close_chat,
        patch.object(llm.embeddings_client, "close", new=AsyncMock()) as close_embeddings,
    ):
        await llm.aclose()

    close_chat.assert_awaited_once()
    close_embeddings.assert_awaited_once()


@pytest.mark.asyncio
async def test_xai_closes_and_forgets_the_client_for_the_running_loop() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    xai_client = MagicMock(close=AsyncMock())
    with patch("any_llm.providers.xai.xai.XaiAsyncClient", return_value=xai_client):
        llm = XaiProvider(api_key="test-api-key")
        assert llm.client is xai_client

        await llm.aclose()

    xai_client.close.assert_awaited_once()
    assert llm._clients_by_loop == {}


class _ChatCompletionHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so the connection stays in the client's pool

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        encoded = json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                ],
            }
        ).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    @override
    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture
def local_api_base() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatCompletionHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_provider_closed_on_its_loop_is_collected_quietly_on_a_later_loop(local_api_base: str) -> None:
    """Regression test for #1439, driving the real OpenAI SDK and httpx against a local server.

    A connection opened on one loop used to stay in the client's pool until the garbage
    collector freed the client, possibly while another loop was running. The SDK then scheduled
    the pool's shutdown on that loop and it failed with ``RuntimeError: Event loop is closed``.
    """
    # Collect what earlier tests left behind while no loop is running, so their clients have no
    # loop to schedule a shutdown on and only this test's provider is collected on the later loop.
    gc.collect()
    llm: AnyLLM | None = AnyLLM.create_openai_compatible("local", api_base=local_api_base, api_key="test")
    reported: list[dict[str, Any]] = []

    async def call_and_close() -> None:
        assert llm is not None
        async with llm:
            await llm.acompletion(model="test-model", messages=[{"role": "user", "content": "Hi"}])

    async def collect_on_a_later_loop() -> None:
        nonlocal llm
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: reported.append(context))
        llm = None
        gc.collect()
        await asyncio.sleep(0.1)

    asyncio.run(call_and_close())
    asyncio.run(collect_on_a_later_loop())
    gc.collect()

    assert reported == []
