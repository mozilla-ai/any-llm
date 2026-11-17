import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from any_llm import AnyLLM
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.tinfoil import TinfoilProvider

ENCLAVE = "router-0.tinfoil.sh"


def _secure_client_mock(http_client: httpx.AsyncClient | None = None) -> MagicMock:
    secure_client = MagicMock()
    secure_client.enclave = ENCLAVE
    secure_client.make_secure_async_http_client.return_value = http_client or httpx.AsyncClient()
    return secure_client


def test_default_client_targets_the_discovered_enclave_through_the_attested_http_client() -> None:
    secure_client = _secure_client_mock()
    with patch("any_llm.providers.tinfoil.tinfoil.SecureClient", return_value=secure_client) as secure_cls:
        provider = TinfoilProvider(api_key="test-key")

    secure_cls.assert_called_once_with(base_url=None)
    assert str(provider.client.base_url) == f"https://{ENCLAVE}/v1/"
    assert provider.client.api_key == "test-key"
    assert provider.client._client is secure_client.make_secure_async_http_client.return_value


def test_api_base_is_used_as_the_request_url_and_passed_to_the_verifier() -> None:
    secure_client = _secure_client_mock()
    with patch("any_llm.providers.tinfoil.tinfoil.SecureClient", return_value=secure_client) as secure_cls:
        provider = TinfoilProvider(api_key="test-key", api_base="https://proxy.example.com/v1/")

    secure_cls.assert_called_once_with(base_url="https://proxy.example.com/v1/")
    assert str(provider.client.base_url) == "https://proxy.example.com/v1/"


def test_api_base_is_read_from_the_environment() -> None:
    secure_client = _secure_client_mock()
    with (
        patch.dict(os.environ, {"TINFOIL_API_KEY": "env-key", "TINFOIL_API_BASE": "https://proxy.example.com/v1/"}),
        patch("any_llm.providers.tinfoil.tinfoil.SecureClient", return_value=secure_client) as secure_cls,
    ):
        provider = TinfoilProvider()

    secure_cls.assert_called_once_with(base_url="https://proxy.example.com/v1/")
    assert provider.client.api_key == "env-key"


def test_verification_options_go_to_the_secure_client_and_the_rest_to_openai() -> None:
    secure_client = _secure_client_mock()
    measurement = {"type": "snp", "registers": ["abc"]}
    with patch("any_llm.providers.tinfoil.tinfoil.SecureClient", return_value=secure_client) as secure_cls:
        provider = TinfoilProvider(
            api_key="test-key",
            enclave="inference.tinfoil.sh",
            repo="tinfoilsh/confidential-model-router",
            measurement=measurement,
            transport="tls",
            attestation_bundle_url="https://bundle.example.com",
            user_cache_secret="secret",  # noqa: S106
            max_retries=5,
            default_headers={"X-Test": "1"},
        )

    secure_cls.assert_called_once_with(
        base_url=None,
        enclave="inference.tinfoil.sh",
        repo="tinfoilsh/confidential-model-router",
        measurement=measurement,
        transport="tls",
        attestation_bundle_url="https://bundle.example.com",
        user_cache_secret="secret",  # noqa: S106
    )
    assert provider.client.max_retries == 5
    assert provider.client.default_headers["X-Test"] == "1"


def test_custom_http_client_is_rejected() -> None:
    with (
        patch("any_llm.providers.tinfoil.tinfoil.SecureClient") as secure_cls,
        pytest.raises(UnsupportedParameterError, match="http_client"),
    ):
        TinfoilProvider(api_key="test-key", http_client=httpx.AsyncClient())

    secure_cls.assert_not_called()


def test_missing_api_key_raises_before_contacting_tinfoil() -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("any_llm.providers.tinfoil.tinfoil.SecureClient") as secure_cls,
        pytest.raises(MissingApiKeyError),
    ):
        TinfoilProvider()

    secure_cls.assert_not_called()


def test_verification_failure_surfaces_from_create() -> None:
    with (
        patch(
            "any_llm.providers.tinfoil.tinfoil.SecureClient",
            side_effect=ValueError("Failed to fetch router addresses"),
        ),
        pytest.raises(ValueError, match="Failed to fetch router addresses"),
    ):
        AnyLLM.create("tinfoil", api_key="test-key", unified_exceptions=False)


@pytest.mark.asyncio
async def test_completion_is_sent_through_the_attested_http_client() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-oss-120b",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
            },
        )

    secure_client = _secure_client_mock(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with patch("any_llm.providers.tinfoil.tinfoil.SecureClient", return_value=secure_client):
        provider = TinfoilProvider(api_key="test-key")

    response: Any = await provider.acompletion(model="gpt-oss-120b", messages=[{"role": "user", "content": "Say hi"}])

    assert response.choices[0].message.content == "hi"
    assert len(requests) == 1
    assert str(requests[0].url) == f"https://{ENCLAVE}/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content)["model"] == "gpt-oss-120b"
