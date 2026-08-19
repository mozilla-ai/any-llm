import pytest

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.registry import get_registry_config, get_registry_provider_class

OvhcloudProvider = get_registry_provider_class("ovhcloud")


def test_provider_metadata() -> None:
    provider = OvhcloudProvider(api_key="test-api-key")
    assert provider.PROVIDER_NAME == "ovhcloud"
    assert provider.API_BASE == "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
    assert provider.ENV_API_KEY_NAME == "OVHCLOUD_API_KEY"
    assert provider.ENV_API_BASE_NAME == "OVHCLOUD_API_BASE"
    assert (
        provider.PROVIDER_DOCUMENTATION_URL
        == "https://docs.ovhcloud.com/en/guides/public-cloud/ai-machine-learning/ai-endpoints-getting-started"
    )


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # OVHcloud AI Endpoints is a hosted, keyed API: constructing without a key must fail.
    monkeypatch.delenv("OVHCLOUD_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        OvhcloudProvider()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVHCLOUD_API_KEY", "env-key")
    provider = OvhcloudProvider()
    assert provider._verify_and_set_api_key(None) == "env-key"


def test_explicit_api_key_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVHCLOUD_API_KEY", "env-key")
    provider = OvhcloudProvider(api_key="explicit-key")
    assert provider._verify_and_set_api_key("explicit-key") == "explicit-key"


def test_api_base_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVHCLOUD_API_BASE", "https://proxy.internal/v1")
    provider = OvhcloudProvider(api_key="test-api-key")
    assert provider._resolve_api_base(None) == "https://proxy.internal/v1"
    assert str(provider.client.base_url).rstrip("/") == "https://proxy.internal/v1"


def test_api_base_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OVHCLOUD_API_BASE", raising=False)
    provider = OvhcloudProvider(api_key="test-api-key")
    assert provider._resolve_api_base(None) is None
    assert str(provider.client.base_url).rstrip("/") == "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"


def test_explicit_api_base_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVHCLOUD_API_BASE", "https://proxy.internal/v1")
    provider = OvhcloudProvider(api_key="test-api-key", api_base="https://explicit.example/v1")
    assert provider._resolve_api_base("https://explicit.example/v1") == "https://explicit.example/v1"
    assert str(provider.client.base_url).rstrip("/") == "https://explicit.example/v1"


def test_capability_flags() -> None:
    # Conservative baseline only: no live key was available to verify anything beyond it.
    assert OvhcloudProvider.SUPPORTS_COMPLETION
    assert OvhcloudProvider.SUPPORTS_COMPLETION_STREAMING
    assert OvhcloudProvider.SUPPORTS_LIST_MODELS
    assert not OvhcloudProvider.SUPPORTS_COMPLETION_REASONING
    assert not OvhcloudProvider.SUPPORTS_COMPLETION_IMAGE
    assert not OvhcloudProvider.SUPPORTS_COMPLETION_PDF
    assert not OvhcloudProvider.SUPPORTS_EMBEDDING
    assert not OvhcloudProvider.SUPPORTS_MODERATION
    assert not OvhcloudProvider.SUPPORTS_BATCH
    assert not OvhcloudProvider.SUPPORTS_IMAGE_GENERATION
    assert not OvhcloudProvider.SUPPORTS_RERANK
    assert not OvhcloudProvider.SUPPORTS_RESPONSES


def test_registered_in_registry_and_loader() -> None:
    assert get_registry_config("ovhcloud") is not None
    assert AnyLLM.get_provider_class("ovhcloud") is OvhcloudProvider
    assert "ovhcloud" in AnyLLM.get_supported_providers()
    # No LLMProvider enum member: this is a brand-new registry-only row, not a
    # migrated provider carrying a legacy folder/enum compat shim.
    assert "ovhcloud" in AnyLLM.get_registry_provider_names()
