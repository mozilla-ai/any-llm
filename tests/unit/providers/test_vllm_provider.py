import httpx
import pytest

from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.providers.vllm.vllm import VllmProvider
from any_llm.types.model import Model


def test_provider_metadata() -> None:
    provider = VllmProvider()
    assert provider.PROVIDER_NAME == "vllm"
    assert provider.API_BASE == "http://localhost:8000/v1"
    assert provider.ENV_API_KEY_NAME == "VLLM_API_KEY"
    assert provider.ENV_API_BASE_NAME == "VLLM_API_BASE"
    assert provider.PROVIDER_DOCUMENTATION_URL == "https://docs.vllm.ai/"


def test_provider_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    provider = VllmProvider()
    assert provider._verify_and_set_api_key(None) == "no-key-required"
    assert provider.client.api_key == "no-key-required"


def test_provider_with_api_key() -> None:
    provider = VllmProvider(api_key="test-api-key")
    assert provider._verify_and_set_api_key("test-api-key") == "test-api-key"
    assert provider.client.api_key == "test-api-key"


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "env-key")
    provider = VllmProvider()
    assert provider._verify_and_set_api_key(None) == "env-key"
    assert provider.client.api_key == "env-key"


def test_explicit_api_key_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "env-key")
    provider = VllmProvider(api_key="explicit-key")
    assert provider._verify_and_set_api_key("explicit-key") == "explicit-key"
    assert provider.client.api_key == "explicit-key"


def test_capability_flags() -> None:
    assert VllmProvider.SUPPORTS_COMPLETION
    assert VllmProvider.SUPPORTS_COMPLETION_STREAMING
    assert VllmProvider.SUPPORTS_COMPLETION_REASONING
    assert VllmProvider.SUPPORTS_COMPLETION_IMAGE
    assert not VllmProvider.SUPPORTS_COMPLETION_PDF
    assert VllmProvider.SUPPORTS_EMBEDDING
    assert VllmProvider.SUPPORTS_MODERATION
    assert VllmProvider.SUPPORTS_LIST_MODELS
    assert not VllmProvider.SUPPORTS_BATCH
    assert not VllmProvider.SUPPORTS_IMAGE_GENERATION
    assert not VllmProvider.SUPPORTS_RERANK
    assert not VllmProvider.SUPPORTS_RESPONSES


def test_registered_in_enum_and_loader() -> None:
    assert LLMProvider.from_string("vllm") is LLMProvider.VLLM
    assert AnyLLM.get_provider_class("vllm") is VllmProvider
    assert "vllm" in AnyLLM.get_supported_providers()


@pytest.mark.asyncio
async def test_list_models_reads_the_served_context_length() -> None:
    # A vLLM ModelCard as ``GET /v1/models`` serves it; ``max_model_len`` is the
    # ``--max-model-len`` the engine was started with, not the model's own maximum.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "Qwen/Qwen3-32B",
                        "object": "model",
                        "created": 1767225600,
                        "owned_by": "vllm",
                        "root": "Qwen/Qwen3-32B",
                        "parent": None,
                        "max_model_len": 32768,
                        "permission": [],
                    }
                ],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = VllmProvider(http_client=http_client)
    try:
        result = await provider.alist_models()
    finally:
        await http_client.aclose()

    assert len(result) == 1
    assert isinstance(result[0], Model)
    assert result[0].id == "Qwen/Qwen3-32B"
    assert result[0].context_window == 32768
    assert result[0].model_extra is not None
    assert result[0].model_extra["max_model_len"] == 32768
