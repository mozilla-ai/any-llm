import pytest

from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.providers.llamacpp.llamacpp import LlamacppProvider


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


def test_registered_in_enum_and_loader() -> None:
    assert LLMProvider.from_string("llamacpp") is LLMProvider.LLAMACPP
    assert AnyLLM.get_provider_class("llamacpp") is LlamacppProvider
    assert "llamacpp" in AnyLLM.get_supported_providers()
