import pytest

from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.registry import get_registry_config, get_registry_provider_class

OuterlabsProvider = get_registry_provider_class("outerlabs")


def test_provider_metadata() -> None:
    provider = OuterlabsProvider(api_key="test-api-key")
    assert provider.PROVIDER_NAME == "outerlabs"
    assert provider.API_BASE == "https://api.outerlabs.ai/v1"
    assert provider.ENV_API_KEY_NAME == "OUTERLABS_API_KEY"
    assert provider.ENV_API_BASE_NAME == "OUTERLABS_API_BASE"
    assert provider.PROVIDER_DOCUMENTATION_URL == "https://outerlabs.ai/docs"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Outer Labs is a hosted, keyed API: constructing without a key must fail.
    monkeypatch.delenv("OUTERLABS_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        OuterlabsProvider()


def test_missing_api_key_does_not_fall_back_to_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The OpenAI SDK reads OPENAI_API_KEY when it is handed no key; that key must never reach this endpoint.
    monkeypatch.delenv("OUTERLABS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    with pytest.raises(MissingApiKeyError) as exc_info:
        OuterlabsProvider()
    assert "OUTERLABS_API_KEY" in str(exc_info.value)


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTERLABS_API_KEY", "env-key")
    provider = OuterlabsProvider()
    assert provider._verify_and_set_api_key(None) == "env-key"


def test_explicit_api_key_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTERLABS_API_KEY", "env-key")
    provider = OuterlabsProvider(api_key="explicit-key")
    assert provider._verify_and_set_api_key("explicit-key") == "explicit-key"


def test_api_base_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTERLABS_API_BASE", "https://proxy.internal/v1")
    provider = OuterlabsProvider(api_key="test-api-key")
    assert provider._resolve_api_base(None) == "https://proxy.internal/v1"
    assert str(provider.client.base_url).rstrip("/") == "https://proxy.internal/v1"


def test_api_base_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTERLABS_API_BASE", raising=False)
    provider = OuterlabsProvider(api_key="test-api-key")
    assert provider._resolve_api_base(None) is None
    assert str(provider.client.base_url).rstrip("/") == "https://api.outerlabs.ai/v1"


def test_explicit_api_base_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTERLABS_API_BASE", "https://proxy.internal/v1")
    provider = OuterlabsProvider(api_key="test-api-key", api_base="https://explicit.example/v1")
    assert provider._resolve_api_base("https://explicit.example/v1") == "https://explicit.example/v1"
    assert str(provider.client.base_url).rstrip("/") == "https://explicit.example/v1"


def test_capability_flags() -> None:
    assert OuterlabsProvider.SUPPORTS_COMPLETION
    assert OuterlabsProvider.SUPPORTS_COMPLETION_STREAMING
    assert OuterlabsProvider.SUPPORTS_LIST_MODELS
    assert not OuterlabsProvider.SUPPORTS_COMPLETION_REASONING
    assert not OuterlabsProvider.SUPPORTS_COMPLETION_IMAGE
    assert not OuterlabsProvider.SUPPORTS_COMPLETION_PDF
    assert not OuterlabsProvider.SUPPORTS_EMBEDDING
    assert not OuterlabsProvider.SUPPORTS_MODERATION
    assert not OuterlabsProvider.SUPPORTS_BATCH
    assert not OuterlabsProvider.SUPPORTS_IMAGE_GENERATION
    assert not OuterlabsProvider.SUPPORTS_RERANK
    assert not OuterlabsProvider.SUPPORTS_RESPONSES


def test_registered_in_registry_and_loader() -> None:
    assert get_registry_config("outerlabs") is not None
    assert AnyLLM.get_provider_class("outerlabs") is OuterlabsProvider
    assert "outerlabs" in AnyLLM.get_supported_providers()
    assert "outerlabs" in AnyLLM.get_registry_provider_names()
    assert LLMProvider("outerlabs") == "outerlabs"
