import pytest

from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.providers.cascadia.cascadia import CascadiaProvider


def test_provider_metadata() -> None:
    provider = CascadiaProvider()
    assert provider.PROVIDER_NAME == "cascadia"
    assert provider.API_BASE == "http://localhost:9090/v1"
    assert provider.ENV_API_KEY_NAME == "CASCADIA_API_KEY"
    assert provider.ENV_API_BASE_NAME == "CASCADIA_API_BASE"
    assert provider.PROVIDER_DOCUMENTATION_URL == "https://cascadia.to"


def test_provider_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keyless coordinators on trusted networks must work (vLLM convention).
    monkeypatch.delenv("CASCADIA_API_KEY", raising=False)
    provider = CascadiaProvider()
    assert provider._verify_and_set_api_key(None) == "no-key-required"


def test_provider_with_api_key() -> None:
    provider = CascadiaProvider(api_key="test-api-key")
    assert provider._verify_and_set_api_key("test-api-key") == "test-api-key"


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A keyed fleet's token set via CASCADIA_API_KEY must be honored.
    monkeypatch.setenv("CASCADIA_API_KEY", "env-key")
    provider = CascadiaProvider()
    assert provider._verify_and_set_api_key(None) == "env-key"


def test_explicit_api_key_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # An explicit key must win over CASCADIA_API_KEY in the environment.
    monkeypatch.setenv("CASCADIA_API_KEY", "env-key")
    provider = CascadiaProvider(api_key="explicit-key")
    assert provider._verify_and_set_api_key("explicit-key") == "explicit-key"


def test_api_base_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASCADIA_API_BASE", "http://fleet.internal:9090/v1")
    provider = CascadiaProvider()
    assert provider._resolve_api_base(None) == "http://fleet.internal:9090/v1"


def test_api_base_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CASCADIA_API_BASE", raising=False)
    provider = CascadiaProvider()
    # No env var, no explicit base: falls through to the class default.
    assert provider._resolve_api_base(None) is None
    assert str(provider.client.base_url).rstrip("/") == "http://localhost:9090/v1"


def test_capability_flags() -> None:
    assert CascadiaProvider.SUPPORTS_COMPLETION
    assert CascadiaProvider.SUPPORTS_COMPLETION_STREAMING
    assert CascadiaProvider.SUPPORTS_COMPLETION_REASONING
    assert CascadiaProvider.SUPPORTS_LIST_MODELS
    assert not CascadiaProvider.SUPPORTS_EMBEDDING
    assert not CascadiaProvider.SUPPORTS_MODERATION
    assert not CascadiaProvider.SUPPORTS_COMPLETION_IMAGE
    assert not CascadiaProvider.SUPPORTS_COMPLETION_PDF
    assert not CascadiaProvider.SUPPORTS_BATCH
    assert not CascadiaProvider.SUPPORTS_IMAGE_GENERATION
    assert not CascadiaProvider.SUPPORTS_RERANK
    assert not CascadiaProvider.SUPPORTS_RESPONSES


def test_registered_in_enum_and_loader() -> None:
    assert LLMProvider.from_string("cascadia") is LLMProvider.CASCADIA
    assert AnyLLM.get_provider_class("cascadia") is CascadiaProvider
    assert "cascadia" in AnyLLM.get_supported_providers()
