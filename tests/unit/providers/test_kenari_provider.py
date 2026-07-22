import pytest

from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.kenari.kenari import KenariProvider


def test_provider_metadata() -> None:
    provider = KenariProvider(api_key="test-api-key")
    assert provider.PROVIDER_NAME == "kenari"
    assert provider.API_BASE == "https://kenari.id/v1"
    assert provider.ENV_API_KEY_NAME == "KENARI_API_KEY"
    assert provider.ENV_API_BASE_NAME == "KENARI_API_BASE"
    assert provider.PROVIDER_DOCUMENTATION_URL == "https://kenari.id/docs"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Kenari is a hosted, keyed API: constructing without a key must fail.
    monkeypatch.delenv("KENARI_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        KenariProvider()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KENARI_API_KEY", "env-key")
    provider = KenariProvider()
    assert provider._verify_and_set_api_key(None) == "env-key"


def test_explicit_api_key_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KENARI_API_KEY", "env-key")
    provider = KenariProvider(api_key="explicit-key")
    assert provider._verify_and_set_api_key("explicit-key") == "explicit-key"


def test_api_base_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KENARI_API_BASE", "https://proxy.internal/v1")
    provider = KenariProvider(api_key="test-api-key")
    assert provider._resolve_api_base(None) == "https://proxy.internal/v1"


def test_api_base_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KENARI_API_BASE", raising=False)
    provider = KenariProvider(api_key="test-api-key")
    # No env var, no explicit base: falls through to the class default.
    assert provider._resolve_api_base(None) is None
    assert str(provider.client.base_url).rstrip("/") == "https://kenari.id/v1"


def test_capability_flags() -> None:
    assert KenariProvider.SUPPORTS_COMPLETION
    assert KenariProvider.SUPPORTS_COMPLETION_STREAMING
    assert KenariProvider.SUPPORTS_COMPLETION_REASONING
    assert KenariProvider.SUPPORTS_LIST_MODELS
    assert not KenariProvider.SUPPORTS_COMPLETION_IMAGE
    assert not KenariProvider.SUPPORTS_COMPLETION_PDF
    assert not KenariProvider.SUPPORTS_EMBEDDING
    assert not KenariProvider.SUPPORTS_MODERATION
    assert not KenariProvider.SUPPORTS_BATCH
    assert not KenariProvider.SUPPORTS_IMAGE_GENERATION
    assert not KenariProvider.SUPPORTS_RERANK
    assert not KenariProvider.SUPPORTS_RESPONSES


def test_registered_in_enum_and_loader() -> None:
    assert LLMProvider.from_string("kenari") is LLMProvider.KENARI
    assert AnyLLM.get_provider_class("kenari") is KenariProvider
    assert "kenari" in AnyLLM.get_supported_providers()
