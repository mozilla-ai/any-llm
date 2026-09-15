import pytest

from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.providers.llmman.llmman import LlmmanProvider


def test_provider_metadata() -> None:
    provider = LlmmanProvider()
    assert provider.PROVIDER_NAME == "llmman"
    assert provider.API_BASE == "http://localhost:17434/v1"
    assert provider.ENV_API_BASE_NAME == "LLMMAN_API_BASE"
    assert provider.PROVIDER_DOCUMENTATION_URL == "https://github.com/llmmanorg/llmman"


def test_provider_without_api_key() -> None:
    provider = LlmmanProvider()
    assert provider._verify_and_set_api_key(None) == "no-key-required"


def test_api_base_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMMAN_API_BASE", raising=False)
    provider = LlmmanProvider()
    assert str(provider.client.base_url).rstrip("/") == "http://localhost:17434/v1"


def test_api_base_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMMAN_API_BASE", "http://gpu-box:17434/v1")
    provider = LlmmanProvider()
    assert str(provider.client.base_url).rstrip("/") == "http://gpu-box:17434/v1"


def test_capability_flags() -> None:
    assert LlmmanProvider.SUPPORTS_COMPLETION
    assert LlmmanProvider.SUPPORTS_COMPLETION_STREAMING
    assert LlmmanProvider.SUPPORTS_COMPLETION_REASONING
    assert LlmmanProvider.SUPPORTS_COMPLETION_IMAGE
    assert LlmmanProvider.SUPPORTS_LIST_MODELS
    assert LlmmanProvider.SUPPORTS_EMBEDDING
    assert not LlmmanProvider.SUPPORTS_COMPLETION_PDF
    assert not LlmmanProvider.SUPPORTS_MODERATION


def test_resolves_by_name_and_enum() -> None:
    assert AnyLLM.get_provider_class("llmman") is LlmmanProvider
    assert AnyLLM.get_provider_class(LLMProvider.LLMMAN) is LlmmanProvider
