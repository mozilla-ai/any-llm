import pytest

from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.registry import get_registry_config, get_registry_provider_class

TokenhubProvider = get_registry_provider_class("tokenhub")


def test_provider_metadata() -> None:
    provider = TokenhubProvider(api_key="test-api-key")
    assert provider.PROVIDER_NAME == "tokenhub"
    assert provider.API_BASE == "https://us-api.tokenhub.com/v1"
    assert provider.ENV_API_KEY_NAME == "TOKENHUB_API_KEY"
    assert provider.ENV_API_BASE_NAME == "TOKENHUB_API_BASE"
    assert provider.PROVIDER_DOCUMENTATION_URL == "https://tokenhub.com/docs"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # TokenHub is a hosted, keyed API: constructing without a key must fail.
    monkeypatch.delenv("TOKENHUB_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        TokenhubProvider()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENHUB_API_KEY", "env-key")
    provider = TokenhubProvider()
    assert provider._verify_and_set_api_key(None) == "env-key"


def test_explicit_api_key_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENHUB_API_KEY", "env-key")
    provider = TokenhubProvider(api_key="explicit-key")
    assert provider._verify_and_set_api_key("explicit-key") == "explicit-key"


def test_api_base_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENHUB_API_BASE", "https://proxy.internal/v1")
    provider = TokenhubProvider(api_key="test-api-key")
    assert provider._resolve_api_base(None) == "https://proxy.internal/v1"
    assert str(provider.client.base_url).rstrip("/") == "https://proxy.internal/v1"


def test_api_base_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKENHUB_API_BASE", raising=False)
    provider = TokenhubProvider(api_key="test-api-key")
    assert provider._resolve_api_base(None) is None
    assert str(provider.client.base_url).rstrip("/") == "https://us-api.tokenhub.com/v1"


def test_explicit_api_base_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENHUB_API_BASE", "https://proxy.internal/v1")
    provider = TokenhubProvider(api_key="test-api-key", api_base="https://explicit.example/v1")
    assert provider._resolve_api_base("https://explicit.example/v1") == "https://explicit.example/v1"
    assert str(provider.client.base_url).rstrip("/") == "https://explicit.example/v1"


def test_capability_flags() -> None:
    assert TokenhubProvider.SUPPORTS_COMPLETION
    assert TokenhubProvider.SUPPORTS_COMPLETION_STREAMING
    assert TokenhubProvider.SUPPORTS_LIST_MODELS
    assert not TokenhubProvider.SUPPORTS_COMPLETION_REASONING
    assert not TokenhubProvider.SUPPORTS_COMPLETION_IMAGE
    assert not TokenhubProvider.SUPPORTS_COMPLETION_PDF
    assert not TokenhubProvider.SUPPORTS_EMBEDDING
    assert not TokenhubProvider.SUPPORTS_MODERATION
    assert not TokenhubProvider.SUPPORTS_BATCH
    assert not TokenhubProvider.SUPPORTS_IMAGE_GENERATION
    assert not TokenhubProvider.SUPPORTS_RERANK
    assert not TokenhubProvider.SUPPORTS_RESPONSES


def test_registered_in_registry_and_loader() -> None:
    assert get_registry_config("tokenhub") is not None
    assert AnyLLM.get_provider_class("tokenhub") is TokenhubProvider
    assert "tokenhub" in AnyLLM.get_supported_providers()
    assert "tokenhub" in AnyLLM.get_registry_provider_names()
    assert LLMProvider("tokenhub") == "tokenhub"
