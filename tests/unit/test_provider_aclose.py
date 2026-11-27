import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typing_extensions import override

from any_llm import AnyLLM
from any_llm.providers.openai import OpenaiProvider
from any_llm.utils.aio import aclose_sdk_client


class _AsyncAcloseClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _AsyncCloseClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _SyncCloseClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _AexitOnlyClient:
    def __init__(self) -> None:
        self.closed = False

    async def __aexit__(self, *exc_info: object) -> None:
        self.closed = True


class _FailingClient:
    async def close(self) -> None:
        msg = "close failed"
        raise RuntimeError(msg)


def _sdk_mock(**methods: Any) -> MagicMock:
    """A stand-in SDK client exposing only the given attributes, as the real SDK does."""
    return MagicMock(spec=list(methods), **methods)


class _ProviderWithClients(OpenaiProvider):
    owned: list[object]

    @override
    def _owned_clients(self) -> list[object]:
        return self.owned


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [_AsyncAcloseClient, _AsyncCloseClient, _SyncCloseClient, _AexitOnlyClient])
async def test_aclose_sdk_client_handles_each_close_spelling(client_class: Any) -> None:
    client = client_class()

    await aclose_sdk_client(client)

    assert client.closed


@pytest.mark.asyncio
async def test_aclose_sdk_client_ignores_client_without_close() -> None:
    await aclose_sdk_client(object())


@pytest.mark.asyncio
async def test_aclose_closes_the_sdk_client() -> None:
    provider = AnyLLM.create("openai", api_key="test-key")
    assert isinstance(provider, OpenaiProvider)
    assert not provider.client.is_closed()

    await provider.aclose()

    assert provider.client.is_closed()


@pytest.mark.asyncio
async def test_async_context_manager_closes_the_sdk_client() -> None:
    async with AnyLLM.create("openai", api_key="test-key") as provider:
        assert isinstance(provider, OpenaiProvider)
        assert not provider.client.is_closed()

    assert provider.client.is_closed()


@pytest.mark.asyncio
async def test_async_context_manager_closes_the_sdk_client_when_the_body_raises() -> None:
    provider = AnyLLM.create("openai", api_key="test-key")
    assert isinstance(provider, OpenaiProvider)

    async def fail_inside() -> None:
        async with provider:
            msg = "boom"
            raise ValueError(msg)

    with pytest.raises(ValueError, match="boom"):
        await fail_inside()

    assert provider.client.is_closed()


def test_aclose_on_the_callers_loop_leaves_nothing_for_a_later_loop() -> None:
    provider = AnyLLM.create("openai", api_key="test-key")
    assert isinstance(provider, OpenaiProvider)

    async def use_and_close() -> None:
        # Open the connection pool on this loop, as a first request would.
        await provider.client._client.__aenter__()
        await provider.aclose()

    asyncio.run(use_and_close())

    assert provider.client.is_closed()


@pytest.mark.asyncio
async def test_aclose_closes_every_client_and_raises_the_first_failure() -> None:
    provider = _ProviderWithClients(api_key="test-key")
    later = _AsyncCloseClient()
    provider.owned = [_FailingClient(), later]

    with pytest.raises(RuntimeError, match="close failed"):
        await provider.aclose()

    assert later.closed


@pytest.mark.asyncio
async def test_aclose_closes_a_shared_client_once() -> None:
    provider = _ProviderWithClients(api_key="test-key")
    shared = MagicMock()
    shared.aclose = AsyncMock()
    provider.owned = [shared, shared]

    await provider.aclose()

    shared.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_aclose_closes_nested_providers() -> None:
    provider = _ProviderWithClients(api_key="test-key")
    nested = AnyLLM.create("openai", api_key="test-key")
    assert isinstance(nested, OpenaiProvider)
    provider.owned = [nested]

    await provider.aclose()

    assert nested.client.is_closed()


@pytest.mark.asyncio
async def test_huggingface_aclose_closes_inference_and_responses_clients() -> None:
    from any_llm.providers.huggingface.huggingface import HuggingfaceProvider

    provider = HuggingfaceProvider(api_key="test-key")

    with patch.object(provider.client, "close", AsyncMock()) as inference_close:
        await provider.aclose()

    inference_close.assert_awaited_once()
    assert provider.responses_client.is_closed()


@pytest.mark.asyncio
async def test_azure_aclose_closes_chat_and_embeddings_clients() -> None:
    from any_llm.providers.azure.azure import AzureProvider

    provider = AzureProvider(api_key="test-key", api_base="https://example.inference.ai.azure.com")
    provider.chat_client = _sdk_mock(close=AsyncMock())
    provider.embeddings_client = _sdk_mock(close=AsyncMock())

    await provider.aclose()

    provider.chat_client.close.assert_awaited_once()
    provider.embeddings_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_gemini_aclose_closes_async_and_sync_transports() -> None:
    from any_llm.providers.gemini.gemini import GeminiProvider

    provider = GeminiProvider(api_key="test-key")
    provider.client = _sdk_mock(close=MagicMock(), aio=_sdk_mock(aclose=AsyncMock()))

    await provider.aclose()

    provider.client.aio.aclose.assert_awaited_once()
    provider.client.close.assert_called_once()


@pytest.mark.asyncio
async def test_vertexai_aclose_closes_partner_and_mistral_clients() -> None:
    from any_llm.providers.vertexai.vertexai import VertexaiProvider

    with patch("any_llm.providers.vertexai.vertexai.genai.Client"):
        provider = VertexaiProvider()
    partner = AnyLLM.create("openai", api_key="test-key")
    mistral_client = _sdk_mock(close=AsyncMock())
    provider._partner_provider = partner  # type: ignore[assignment]
    provider._mistral_client = mistral_client

    await provider.aclose()

    assert isinstance(partner, OpenaiProvider)
    assert partner.client.is_closed()
    mistral_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_bedrock_aclose_leaves_a_caller_supplied_client_open() -> None:
    from any_llm.providers.bedrock.bedrock import BedrockProvider

    caller_client = MagicMock()
    provider = BedrockProvider(client=caller_client)

    await provider.aclose()

    caller_client.close.assert_not_called()


@pytest.mark.asyncio
async def test_bedrock_aclose_closes_the_clients_it_created() -> None:
    from any_llm.providers.bedrock.bedrock import BedrockProvider

    with patch("boto3.Session") as session:
        runtime_client = _sdk_mock(close=MagicMock())
        timeout_client = _sdk_mock(close=MagicMock())
        session.return_value.client.return_value = runtime_client
        provider = BedrockProvider()
        provider._timeout_clients[5.0] = timeout_client

        await provider.aclose()

    runtime_client.close.assert_called_once()
    timeout_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_xai_aclose_closes_the_running_loops_client() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    with patch("any_llm.providers.xai.xai.XaiAsyncClient") as xai_client_class:
        first = _sdk_mock(close=AsyncMock())
        second = _sdk_mock(close=AsyncMock())
        xai_client_class.side_effect = [first, second]
        provider = XaiProvider(api_key="test-key")
        assert provider.client is first

        await provider.aclose()

        first.close.assert_awaited_once()
        assert provider.client is second


@pytest.mark.asyncio
async def test_xai_aclose_without_a_client_on_this_loop_is_a_no_op() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    with patch("any_llm.providers.xai.xai.XaiAsyncClient") as xai_client_class:
        provider = XaiProvider(api_key="test-key")

        await provider.aclose()

    xai_client_class.assert_not_called()
