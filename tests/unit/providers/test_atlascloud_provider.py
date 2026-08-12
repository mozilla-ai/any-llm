import pytest

from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.atlascloud.atlascloud import AtlascloudProvider


def test_provider_metadata() -> None:
    provider = AtlascloudProvider(api_key="test-api-key")
    assert provider.PROVIDER_NAME == "atlascloud"
    assert provider.API_BASE == "https://api.atlascloud.ai/v1"
    assert provider.ENV_API_KEY_NAME == "ATLASCLOUD_API_KEY"
    assert provider.ENV_API_BASE_NAME == "ATLASCLOUD_API_BASE"
    assert provider.PROVIDER_DOCUMENTATION_URL == "https://www.atlascloud.ai/docs"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Atlas Cloud is a hosted, keyed API: constructing without a key must fail.
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        AtlascloudProvider()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "env-key")
    provider = AtlascloudProvider()
    assert provider._verify_and_set_api_key(None) == "env-key"


def test_explicit_api_key_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "env-key")
    provider = AtlascloudProvider(api_key="explicit-key")
    assert provider._verify_and_set_api_key("explicit-key") == "explicit-key"


def test_api_base_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLASCLOUD_API_BASE", "https://proxy.internal/v1")
    provider = AtlascloudProvider(api_key="test-api-key")
    assert provider._resolve_api_base(None) == "https://proxy.internal/v1"


def test_api_base_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLASCLOUD_API_BASE", raising=False)
    provider = AtlascloudProvider(api_key="test-api-key")
    # No env var, no explicit base: falls through to the class default.
    assert provider._resolve_api_base(None) is None
    assert str(provider.client.base_url).rstrip("/") == "https://api.atlascloud.ai/v1"


def test_capability_flags() -> None:
    assert AtlascloudProvider.SUPPORTS_COMPLETION
    assert AtlascloudProvider.SUPPORTS_COMPLETION_STREAMING
    assert AtlascloudProvider.SUPPORTS_COMPLETION_REASONING
    assert AtlascloudProvider.SUPPORTS_LIST_MODELS
    assert not AtlascloudProvider.SUPPORTS_COMPLETION_IMAGE
    assert not AtlascloudProvider.SUPPORTS_COMPLETION_PDF
    assert not AtlascloudProvider.SUPPORTS_EMBEDDING
    assert not AtlascloudProvider.SUPPORTS_MODERATION
    assert not AtlascloudProvider.SUPPORTS_BATCH
    assert not AtlascloudProvider.SUPPORTS_IMAGE_GENERATION
    assert not AtlascloudProvider.SUPPORTS_RERANK
    assert not AtlascloudProvider.SUPPORTS_RESPONSES


def test_registered_in_enum_and_loader() -> None:
    assert LLMProvider.from_string("atlascloud") is LLMProvider.ATLASCLOUD
    assert AnyLLM.get_provider_class("atlascloud") is AtlascloudProvider
    assert "atlascloud" in AnyLLM.get_supported_providers()
