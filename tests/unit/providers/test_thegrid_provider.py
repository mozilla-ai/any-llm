import pytest

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.registry import get_registry_config, get_registry_provider_class

ThegridProvider = get_registry_provider_class("thegrid")


def test_provider_metadata() -> None:
    provider = ThegridProvider(api_key="test-api-key")
    assert provider.PROVIDER_NAME == "thegrid"
    assert provider.API_BASE == "https://api.thegrid.ai/v1"
    assert provider.ENV_API_KEY_NAME == "THEGRID_API_KEY"
    assert provider.ENV_API_BASE_NAME == "THEGRID_API_BASE"
    assert provider.PROVIDER_DOCUMENTATION_URL == "https://thegrid.ai/docs"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Grid is a hosted, keyed API: constructing without a key must fail.
    monkeypatch.delenv("THEGRID_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        ThegridProvider()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THEGRID_API_KEY", "env-key")
    provider = ThegridProvider()
    assert provider._verify_and_set_api_key(None) == "env-key"


def test_explicit_api_key_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THEGRID_API_KEY", "env-key")
    provider = ThegridProvider(api_key="explicit-key")
    assert provider._verify_and_set_api_key("explicit-key") == "explicit-key"


def test_api_base_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THEGRID_API_BASE", "https://proxy.internal/v1")
    provider = ThegridProvider(api_key="test-api-key")
    assert provider._resolve_api_base(None) == "https://proxy.internal/v1"
    assert str(provider.client.base_url).rstrip("/") == "https://proxy.internal/v1"


def test_api_base_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THEGRID_API_BASE", raising=False)
    provider = ThegridProvider(api_key="test-api-key")
    assert provider._resolve_api_base(None) is None
    assert str(provider.client.base_url).rstrip("/") == "https://api.thegrid.ai/v1"


def test_explicit_api_base_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THEGRID_API_BASE", "https://proxy.internal/v1")
    provider = ThegridProvider(api_key="test-api-key", api_base="https://explicit.example/v1")
    assert provider._resolve_api_base("https://explicit.example/v1") == "https://explicit.example/v1"
    assert str(provider.client.base_url).rstrip("/") == "https://explicit.example/v1"


def test_capability_flags() -> None:
    # Every flag enabled below was exercised against the live API; see the PR
    # description for the transcript. Everything else stays at the conservative
    # gateway baseline.
    assert ThegridProvider.SUPPORTS_COMPLETION
    assert ThegridProvider.SUPPORTS_COMPLETION_STREAMING
    assert ThegridProvider.SUPPORTS_LIST_MODELS
    assert ThegridProvider.SUPPORTS_COMPLETION_REASONING
    assert ThegridProvider.SUPPORTS_COMPLETION_IMAGE
    assert ThegridProvider.SUPPORTS_RESPONSES
    assert not ThegridProvider.SUPPORTS_COMPLETION_PDF
    assert not ThegridProvider.SUPPORTS_EMBEDDING
    assert not ThegridProvider.SUPPORTS_MODERATION
    assert not ThegridProvider.SUPPORTS_BATCH
    assert not ThegridProvider.SUPPORTS_IMAGE_GENERATION
    assert not ThegridProvider.SUPPORTS_RERANK


def test_registered_in_registry_and_loader() -> None:
    assert get_registry_config("thegrid") is not None
    assert AnyLLM.get_provider_class("thegrid") is ThegridProvider
    assert "thegrid" in AnyLLM.get_supported_providers()
    # No LLMProvider enum member: this is a brand-new registry-only row, not a
    # migrated provider carrying a legacy folder/enum compat shim.
    assert "thegrid" in AnyLLM.get_registry_provider_names()
