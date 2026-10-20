import json

import httpx
import pytest

from any_llm import adecide
from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import AnyLLMError
from any_llm.providers.llamacpp.llamacpp import LlamacppProvider
from any_llm.types.decide import Noul


def test_provider_metadata() -> None:
    provider = LlamacppProvider()
    assert provider.PROVIDER_NAME == "llamacpp"
    assert provider.API_BASE == "http://127.0.0.1:8080/v1"
    assert provider.ENV_API_KEY_NAME == "LLAMACPP_API_KEY"
    assert provider.ENV_API_BASE_NAME == "LLAMACPP_API_BASE"
    assert provider.PROVIDER_DOCUMENTATION_URL == "https://github.com/ggml-org/llama.cpp"


def test_provider_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLAMACPP_API_KEY", raising=False)
    provider = LlamacppProvider()
    assert provider._verify_and_set_api_key(None) == "no-key-required"
    assert provider.client.api_key == "no-key-required"


def test_provider_with_api_key() -> None:
    provider = LlamacppProvider(api_key="test-api-key")
    assert provider._verify_and_set_api_key("test-api-key") == "test-api-key"
    assert provider.client.api_key == "test-api-key"


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMACPP_API_KEY", "env-key")
    provider = LlamacppProvider()
    assert provider._verify_and_set_api_key(None) == "env-key"
    assert provider.client.api_key == "env-key"


def test_explicit_api_key_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMACPP_API_KEY", "env-key")
    provider = LlamacppProvider(api_key="explicit-key")
    assert provider._verify_and_set_api_key("explicit-key") == "explicit-key"
    assert provider.client.api_key == "explicit-key"


def test_capability_flags() -> None:
    assert LlamacppProvider.SUPPORTS_COMPLETION
    assert LlamacppProvider.SUPPORTS_COMPLETION_STREAMING
    assert LlamacppProvider.SUPPORTS_COMPLETION_REASONING
    assert LlamacppProvider.SUPPORTS_COMPLETION_IMAGE
    assert not LlamacppProvider.SUPPORTS_COMPLETION_PDF
    assert LlamacppProvider.SUPPORTS_EMBEDDING
    assert LlamacppProvider.SUPPORTS_MODERATION
    assert LlamacppProvider.SUPPORTS_LIST_MODELS
    assert not LlamacppProvider.SUPPORTS_BATCH
    assert not LlamacppProvider.SUPPORTS_IMAGE_GENERATION
    assert not LlamacppProvider.SUPPORTS_RERANK
    assert not LlamacppProvider.SUPPORTS_RESPONSES
    assert LlamacppProvider.SUPPORTS_DECIDE
    assert LlamacppProvider.SUPPORTS_DECIDE_IMAGE


def test_registered_in_enum_and_loader() -> None:
    assert LLMProvider.from_string("llamacpp") is LLMProvider.LLAMACPP
    assert AnyLLM.get_provider_class("llamacpp") is LlamacppProvider
    assert "llamacpp" in AnyLLM.get_supported_providers()


def _answering(requests: list[httpx.Request]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = {"model": "tev1", "answers": {"spam": {"type": "noul", "noul": 0.2}}, "usage": {"input_tokens": 5}}
        return httpx.Response(200, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_adecide_posts_to_systemone_under_the_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLAMACPP_API_KEY", raising=False)
    requests: list[httpx.Request] = []
    provider = LlamacppProvider(http_client=_answering(requests))

    result = await provider.adecide("tev1", "Buy now!", {"spam": Noul(instructions="Is this spam?")})

    (request,) = requests
    assert str(request.url) == "http://127.0.0.1:8080/v1/systemone"
    assert json.loads(request.content) == {
        "model": "tev1",
        "state": "Buy now!",
        "questions": {"spam": {"type": "noul", "instructions": "Is this spam?"}},
    }
    assert result.nouls["spam"].noul == 0.2


@pytest.mark.asyncio
async def test_top_level_adecide_dispatches_on_the_llamacpp_prefix() -> None:
    requests: list[httpx.Request] = []

    result = await adecide(
        "llamacpp:tev1",
        "Buy now!",
        {"spam": Noul(instructions="Is this spam?")},
        api_base="http://gpu-box:8080/v1",
        client_args={"http_client": _answering(requests)},
    )

    assert str(requests[0].url) == "http://gpu-box:8080/v1/systemone"
    assert result.model == "tev1"


@pytest.mark.asyncio
async def test_adecide_sends_images_in_the_body() -> None:
    requests: list[httpx.Request] = []
    provider = LlamacppProvider(http_client=_answering(requests))
    images = ["data:image/png;base64,iVBORw0KGgo="]

    await provider.adecide("laya", "What is in the picture?", {"spam": Noul(instructions="Is it spam?")}, images=images)

    assert json.loads(requests[0].content)["images"] == images


@pytest.mark.asyncio
async def test_adecide_sends_one_request_with_max_retries_zero() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "Loading model"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = LlamacppProvider(max_retries=0, unified_exceptions=True, http_client=http_client)

    with pytest.raises(AnyLLMError) as exc_info:
        await provider.adecide("laya", "state", {"spam": Noul(instructions="Is it spam?")})

    assert exc_info.value.status_code == 503
    assert len(requests) == 1
