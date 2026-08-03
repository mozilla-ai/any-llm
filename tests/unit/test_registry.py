from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.any_llm import AnyLLM
from any_llm.api import (
    acancel_batch,
    acreate_batch,
    alist_batches,
    alist_models,
    aretrieve_batch,
    aretrieve_batch_results,
    cancel_batch,
    completion,
    create_batch,
    embedding,
    image_generation,
    list_batches,
    list_models,
    moderation,
    retrieve_batch,
    retrieve_batch_results,
)
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


def test_registry_only_name_is_listed_as_supported(community_row: OpenAICompatibleProviderConfig) -> None:
    assert "testgateway" in AnyLLM.get_supported_providers()
    assert "testgateway" in AnyLLM.get_registry_provider_names()


def test_registry_provider_names_excludes_rows_that_have_an_enum_member() -> None:
    # atlascloud is a row and an enum member, so it is already covered by the enum listing.
    assert "atlascloud" not in AnyLLM.get_registry_provider_names()
    assert "atlascloud" in AnyLLM.get_supported_providers()


def test_registry_only_name_appears_in_metadata(community_row: OpenAICompatibleProviderConfig) -> None:
    metadata = {entry.name: entry for entry in AnyLLM.get_all_provider_metadata()}
    assert metadata["testgateway"].env_key == "TESTGATEWAY_API_KEY"
    assert metadata["testgateway"].class_name == "TestgatewayProvider"


def test_resolve_provider_key_returns_enum_member_when_one_exists() -> None:
    assert AnyLLM.resolve_provider_key("openai") is LLMProvider.OPENAI
    assert AnyLLM.resolve_provider_key(LLMProvider.OPENAI) is LLMProvider.OPENAI


def test_resolve_provider_key_returns_bare_name_for_registry_only_row(
    community_row: OpenAICompatibleProviderConfig,
) -> None:
    assert AnyLLM.resolve_provider_key("testgateway") == "testgateway"


def test_resolve_provider_key_normalizes_like_from_string(community_row: OpenAICompatibleProviderConfig) -> None:
    assert AnyLLM.resolve_provider_key("  TestGateway ") == "testgateway"
    assert AnyLLM.resolve_provider_key("  OpenAI ") is LLMProvider.OPENAI


def test_resolve_provider_key_unknown_name_lists_registry_names(
    community_row: OpenAICompatibleProviderConfig,
) -> None:
    with pytest.raises(UnsupportedProviderError) as excinfo:
        AnyLLM.resolve_provider_key("not-a-provider")
    assert "testgateway" in str(excinfo.value)


def test_string_routing_resolves_registry_only_row(community_row: OpenAICompatibleProviderConfig) -> None:
    provider_key, model_id = AnyLLM.split_model_provider("testgateway:some-model")
    assert provider_key == "testgateway"
    assert model_id == "some-model"
    assert AnyLLM.get_provider_class(provider_key) is get_registry_provider_class("testgateway")


def test_legacy_slash_routing_resolves_registry_only_row(community_row: OpenAICompatibleProviderConfig) -> None:
    with pytest.deprecated_call():
        provider_key, model_id = AnyLLM.split_model_provider("testgateway/some-model")
    assert provider_key == "testgateway"
    assert model_id == "some-model"


def test_split_model_provider_still_returns_enum_for_enum_providers() -> None:
    provider_key, model_id = AnyLLM.split_model_provider("openai:gpt-4o")
    assert provider_key is LLMProvider.OPENAI
    assert model_id == "gpt-4o"


def test_api_provider_argument_accepts_registry_only_row(community_row: OpenAICompatibleProviderConfig) -> None:
    """The explicit provider= argument resolves registry rows, not just the enum."""
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        completion(model="some-model", provider="testgateway", messages=[{"role": "user", "content": "Hello"}])

    assert mock_create.call_args.args[0] == "testgateway"
    mock_provider.completion.assert_called_once()


def test_api_string_routing_accepts_registry_only_row(community_row: OpenAICompatibleProviderConfig) -> None:
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        completion(model="testgateway:some-model", messages=[{"role": "user", "content": "Hello"}])

    assert mock_create.call_args.args[0] == "testgateway"
    mock_provider.completion.assert_called_once()


def test_unsupported_provider_error_lists_registry_names_from_get_provider_enum(
    community_row: OpenAICompatibleProviderConfig,
) -> None:
    """get_provider_enum's error must not omit registry-only gateways.

    Regression guard: its supported list was the enum alone, which diverged from
    get_supported_providers() the moment a row without an enum member existed.
    """
    with pytest.raises(UnsupportedProviderError) as excinfo:
        AnyLLM.get_provider_enum("nonexistent")
    assert excinfo.value.supported_providers == AnyLLM.get_supported_providers()
    assert "testgateway" in excinfo.value.supported_providers


def test_get_provider_enum_still_rejects_registry_only_names(
    community_row: OpenAICompatibleProviderConfig,
) -> None:
    """A registry row has no enum member, so the enum accessor raises by design."""
    with pytest.raises(UnsupportedProviderError):
        AnyLLM.get_provider_enum("testgateway")
    # ...while the resolver accepts it.
    assert AnyLLM.resolve_provider_key("testgateway") == "testgateway"


def test_create_unsupported_error_lists_registry_names(community_row: OpenAICompatibleProviderConfig) -> None:
    """create() must report registry gateways in its supported list.

    Regression guard: create() and get_provider_class() resolved through
    LLMProvider.from_string, so their error listed only the enum even after
    get_provider_enum was fixed.
    """
    with pytest.raises(UnsupportedProviderError) as excinfo:
        AnyLLM.create("not-a-provider", api_key="k")
    assert excinfo.value.supported_providers == AnyLLM.get_supported_providers()
    assert "testgateway" in excinfo.value.supported_providers


def test_get_provider_class_unsupported_error_lists_registry_names(
    community_row: OpenAICompatibleProviderConfig,
) -> None:
    with pytest.raises(UnsupportedProviderError) as excinfo:
        AnyLLM.get_provider_class("not-a-provider")
    assert excinfo.value.supported_providers == AnyLLM.get_supported_providers()
    assert "testgateway" in excinfo.value.supported_providers


@pytest.mark.parametrize(
    ("api_function", "call_kwargs"),
    [
        (completion, {"messages": [{"role": "user", "content": "hi"}]}),
        (embedding, {"inputs": "hi"}),
        (moderation, {"input": "hi"}),
        (image_generation, {"prompt": "a cat"}),
    ],
    ids=["completion", "embedding", "moderation", "image_generation"],
)
def test_api_entry_points_resolve_registry_only_rows(
    community_row: OpenAICompatibleProviderConfig,
    api_function: Any,
    call_kwargs: dict[str, Any],
) -> None:
    """Every api.py entry point resolves a registry-only name, not just completion."""
    mock_provider = Mock()

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        api_function(model="some-model", provider="testgateway", **call_kwargs)

    assert mock_create.call_args.args[0] == "testgateway"


@pytest.mark.parametrize(
    ("api_function", "call_kwargs"),
    [
        (completion, {"messages": [{"role": "user", "content": "hi"}]}),
        (embedding, {"inputs": "hi"}),
        (moderation, {"input": "hi"}),
        (image_generation, {"prompt": "a cat"}),
    ],
    ids=["completion", "embedding", "moderation", "image_generation"],
)
def test_api_entry_points_still_resolve_enum_providers(api_function: Any, call_kwargs: dict[str, Any]) -> None:
    mock_provider = Mock()

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        api_function(model="some-model", provider="openai", **call_kwargs)

    assert mock_create.call_args.args[0] is LLMProvider.OPENAI


# The provider-only entry points take no model string, so a registry row is only
# reachable through the explicit provider= argument.
PROVIDER_ONLY_ENTRY_POINTS = [
    (list_models, {}),
    (create_batch, {"input_file_path": "batch.jsonl", "endpoint": "/v1/chat/completions"}),
    (retrieve_batch, {"batch_id": "batch-1"}),
    (cancel_batch, {"batch_id": "batch-1"}),
    (list_batches, {}),
    (retrieve_batch_results, {"batch_id": "batch-1"}),
]
PROVIDER_ONLY_IDS = ["list_models", "create_batch", "retrieve_batch", "cancel_batch", "list_batches", "batch_results"]


@pytest.mark.parametrize(("api_function", "call_kwargs"), PROVIDER_ONLY_ENTRY_POINTS, ids=PROVIDER_ONLY_IDS)
def test_provider_only_entry_points_resolve_registry_only_rows(
    community_row: OpenAICompatibleProviderConfig,
    api_function: Any,
    call_kwargs: dict[str, Any],
) -> None:
    """list_models and the batch helpers take provider= without a model string.

    They resolved through LLMProvider.from_string, so a registry-only gateway was
    rejected even though get_supported_providers() advertised it.
    """
    mock_provider = Mock()

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        api_function(provider="testgateway", **call_kwargs)

    assert mock_create.call_args.args[0] == "testgateway"


@pytest.mark.parametrize(("api_function", "call_kwargs"), PROVIDER_ONLY_ENTRY_POINTS, ids=PROVIDER_ONLY_IDS)
def test_provider_only_entry_points_still_resolve_enum_providers(
    api_function: Any, call_kwargs: dict[str, Any]
) -> None:
    mock_provider = Mock()

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        api_function(provider="openai", **call_kwargs)

    assert mock_create.call_args.args[0] is LLMProvider.OPENAI


@pytest.mark.parametrize(("api_function", "call_kwargs"), PROVIDER_ONLY_ENTRY_POINTS, ids=PROVIDER_ONLY_IDS)
def test_provider_only_entry_points_report_registry_names_when_unresolvable(
    community_row: OpenAICompatibleProviderConfig,
    api_function: Any,
    call_kwargs: dict[str, Any],
) -> None:
    with pytest.raises(UnsupportedProviderError) as excinfo:
        api_function(provider="not-a-provider", **call_kwargs)
    assert excinfo.value.supported_providers == AnyLLM.get_supported_providers()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_function", "call_kwargs"),
    [
        (alist_models, {}),
        (acreate_batch, {"input_file_path": "batch.jsonl", "endpoint": "/v1/chat/completions"}),
        (aretrieve_batch, {"batch_id": "batch-1"}),
        (acancel_batch, {"batch_id": "batch-1"}),
        (alist_batches, {}),
        (aretrieve_batch_results, {"batch_id": "batch-1"}),
    ],
    ids=PROVIDER_ONLY_IDS,
)
async def test_async_provider_only_entry_points_resolve_registry_only_rows(
    community_row: OpenAICompatibleProviderConfig,
    api_function: Any,
    call_kwargs: dict[str, Any],
) -> None:
    mock_provider = AsyncMock()

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        await api_function(provider="testgateway", **call_kwargs)

    assert mock_create.call_args.args[0] == "testgateway"
