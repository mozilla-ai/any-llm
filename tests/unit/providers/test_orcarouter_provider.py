import pytest

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.registry import get_registry_config, get_registry_provider_class

OrcarouterProvider = get_registry_provider_class("orcarouter")


def test_provider_metadata() -> None:
    provider = OrcarouterProvider(api_key="test-api-key")
    assert provider.PROVIDER_NAME == "orcarouter"
    assert provider.API_BASE == "https://api.orcarouter.ai/v1"
    assert provider.ENV_API_KEY_NAME == "ORCAROUTER_API_KEY"
    assert provider.ENV_API_BASE_NAME == "ORCAROUTER_API_BASE"
    assert provider.PROVIDER_DOCUMENTATION_URL == "https://docs.orcarouter.ai"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # OrcaRouter is a hosted, keyed gateway: constructing without a key must fail.
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        OrcarouterProvider()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "env-key")
    provider = OrcarouterProvider()
    assert provider._verify_and_set_api_key(None) == "env-key"


def test_explicit_api_key_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "env-key")
    provider = OrcarouterProvider(api_key="explicit-key")
    assert provider._verify_and_set_api_key("explicit-key") == "explicit-key"


def test_api_base_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_BASE", "https://proxy.internal/v1")
    provider = OrcarouterProvider(api_key="test-api-key")
    assert provider._resolve_api_base(None) == "https://proxy.internal/v1"
    assert str(provider.client.base_url).rstrip("/") == "https://proxy.internal/v1"


def test_api_base_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORCAROUTER_API_BASE", raising=False)
    provider = OrcarouterProvider(api_key="test-api-key")
    assert provider._resolve_api_base(None) is None
    assert str(provider.client.base_url).rstrip("/") == "https://api.orcarouter.ai/v1"


def test_explicit_api_base_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_BASE", "https://proxy.internal/v1")
    provider = OrcarouterProvider(api_key="test-api-key", api_base="https://explicit.example/v1")
    assert provider._resolve_api_base("https://explicit.example/v1") == "https://explicit.example/v1"
    assert str(provider.client.base_url).rstrip("/") == "https://explicit.example/v1"


def test_capability_flags() -> None:
    # Conservative baseline only: completion, streaming and model listing were
    # exercised against the live endpoint; everything else stays opt-in.
    assert OrcarouterProvider.SUPPORTS_COMPLETION
    assert OrcarouterProvider.SUPPORTS_COMPLETION_STREAMING
    assert OrcarouterProvider.SUPPORTS_LIST_MODELS
    assert not OrcarouterProvider.SUPPORTS_COMPLETION_REASONING
    assert not OrcarouterProvider.SUPPORTS_COMPLETION_IMAGE
    assert not OrcarouterProvider.SUPPORTS_COMPLETION_PDF
    assert not OrcarouterProvider.SUPPORTS_EMBEDDING
    assert not OrcarouterProvider.SUPPORTS_MODERATION
    assert not OrcarouterProvider.SUPPORTS_BATCH
    assert not OrcarouterProvider.SUPPORTS_IMAGE_GENERATION
    assert not OrcarouterProvider.SUPPORTS_RERANK
    assert not OrcarouterProvider.SUPPORTS_RESPONSES


def test_registered_in_registry_and_loader() -> None:
    assert get_registry_config("orcarouter") is not None
    assert AnyLLM.get_provider_class("orcarouter") is OrcarouterProvider
    assert "orcarouter" in AnyLLM.get_supported_providers()
    # No LLMProvider enum member: this is a brand-new registry-only row, not a
    # migrated provider carrying a legacy folder/enum compat shim.
    assert "orcarouter" in AnyLLM.get_registry_provider_names()
