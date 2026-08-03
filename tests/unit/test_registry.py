from collections.abc import Generator

import pytest

from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import UnsupportedProviderError
from any_llm.providers import registry
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.registry import (
    OpenAICompatibleProviderConfig,
    get_registry_config,
    get_registry_provider_class,
)


@pytest.fixture
def community_row(monkeypatch: pytest.MonkeyPatch) -> Generator[OpenAICompatibleProviderConfig, None, None]:
    """Inject a registry row for a gateway that has no enum entry and no folder."""
    config = OpenAICompatibleProviderConfig(
        name="testgateway",
        api_base="https://testgateway.example/v1",
        env_api_key_name="TESTGATEWAY_API_KEY",
        provider_documentation_url="https://testgateway.example/docs",
    )
    monkeypatch.setitem(registry.PROVIDER_REGISTRY, "testgateway", config)
    yield config
    registry._class_cache.pop("testgateway", None)


def test_config_lookup_normalizes_name() -> None:
    assert get_registry_config(" AtlasCloud ") is get_registry_config("atlascloud")


def test_config_lookup_unknown_name_returns_none() -> None:
    assert get_registry_config("not-registered") is None


def test_provider_class_is_cached() -> None:
    assert get_registry_provider_class("atlascloud") is get_registry_provider_class("atlascloud")


def test_provider_class_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        get_registry_provider_class("not-registered")


def test_loader_resolves_registry_row() -> None:
    cls = AnyLLM.get_provider_class("atlascloud")
    assert cls is get_registry_provider_class("atlascloud")
    assert issubclass(cls, BaseOpenAIProvider)


def test_loader_resolves_enum_member_via_registry() -> None:
    assert AnyLLM.get_provider_class(LLMProvider.ATLASCLOUD) is get_registry_provider_class("atlascloud")


def test_create_returns_registry_provider_instance() -> None:
    provider = AnyLLM.create("atlascloud", api_key="test-key")
    assert isinstance(provider, BaseOpenAIProvider)
    assert provider.PROVIDER_NAME == "atlascloud"
    assert str(provider.client.base_url).rstrip("/") == "https://api.atlascloud.ai/v1"


def test_metadata_matches_row() -> None:
    metadata = AnyLLM.get_provider_class("atlascloud").get_provider_metadata()
    assert metadata.name == "atlascloud"
    assert metadata.class_name == "AtlascloudProvider"
    assert metadata.env_key == "ATLASCLOUD_API_KEY"
    assert metadata.env_api_base == "ATLASCLOUD_API_BASE"
    assert metadata.doc_url == "https://www.atlascloud.ai/docs"
    assert metadata.reasoning
    assert not metadata.image


def test_migrated_provider_appears_in_all_provider_metadata() -> None:
    all_metadata = AnyLLM.get_all_provider_metadata()
    assert any(m.name == "atlascloud" for m in all_metadata)


def test_registry_only_name_resolves_without_enum_entry(community_row: OpenAICompatibleProviderConfig) -> None:
    provider = AnyLLM.create("testgateway", api_key="k")
    assert isinstance(provider, BaseOpenAIProvider)
    assert provider.PROVIDER_NAME == "testgateway"
    assert str(provider.client.base_url).rstrip("/") == "https://testgateway.example/v1"
    with pytest.raises(UnsupportedProviderError):
        LLMProvider.from_string("testgateway")


def test_row_defaults_are_conservative(community_row: OpenAICompatibleProviderConfig) -> None:
    cls = get_registry_provider_class("testgateway")
    assert cls.SUPPORTS_COMPLETION
    assert cls.SUPPORTS_COMPLETION_STREAMING
    assert cls.SUPPORTS_LIST_MODELS
    assert not cls.SUPPORTS_COMPLETION_REASONING
    assert not cls.SUPPORTS_COMPLETION_IMAGE
    assert not cls.SUPPORTS_COMPLETION_PDF
    assert not cls.SUPPORTS_EMBEDDING
    assert not cls.SUPPORTS_MODERATION
    assert not cls.SUPPORTS_RESPONSES
    assert not cls.SUPPORTS_BATCH
    assert not cls.SUPPORTS_IMAGE_GENERATION
    assert not cls.SUPPORTS_RERANK


def test_unregistered_name_still_raises_unsupported_provider() -> None:
    with pytest.raises(UnsupportedProviderError):
        AnyLLM.create("not-a-provider", api_key="k")


def test_import_shim_returns_registry_class() -> None:
    from any_llm.providers.atlascloud import AtlascloudProvider
    from any_llm.providers.atlascloud.atlascloud import AtlascloudProvider as DeepAtlascloudProvider

    assert AtlascloudProvider is get_registry_provider_class("atlascloud")
    assert DeepAtlascloudProvider is AtlascloudProvider
