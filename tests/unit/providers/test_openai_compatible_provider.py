import pytest

from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import UnsupportedProviderError
from any_llm.providers.openai.custom import OpenAICompatibleProvider


def test_factory_returns_openai_compatible_provider() -> None:
    provider = AnyLLM.create_openai_compatible(name="mygateway", api_base="https://mygateway.example/v1")
    assert isinstance(provider, OpenAICompatibleProvider)


def test_reports_custom_identity_not_openai() -> None:
    provider = AnyLLM.create_openai_compatible(name="mygateway", api_base="https://mygateway.example/v1")
    assert provider.PROVIDER_NAME == "mygateway"


def test_metadata_reports_custom_identity() -> None:
    # get_provider_metadata is a classmethod reading cls.PROVIDER_NAME; the per-name
    # subclass must make it report the caller's name, not the generic default.
    provider = AnyLLM.create_openai_compatible(name="mygateway", api_base="https://mygateway.example/v1")
    metadata = provider.get_provider_metadata()
    assert metadata.name == "mygateway"
    assert metadata.class_name == "OpenAICompatibleProvider"


def test_api_base_is_bound_to_client() -> None:
    provider = OpenAICompatibleProvider(api_base="https://mygateway.example/v1")
    assert str(provider.client.base_url).rstrip("/") == "https://mygateway.example/v1"


def test_empty_api_base_raises() -> None:
    with pytest.raises(ValueError, match="api_base"):
        AnyLLM.create_openai_compatible(name="mygateway", api_base="")


@pytest.mark.parametrize("name", ["", "   "])
def test_blank_name_raises(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        AnyLLM.create_openai_compatible(name=name, api_base="https://mygateway.example/v1")


def test_keyless_endpoint_uses_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    # A keyless local server must not be blocked by MissingApiKeyError.
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(api_base="http://localhost:8000/v1")
    assert provider.client.api_key == "no-key-required"


def test_explicit_api_key_used() -> None:
    provider = OpenAICompatibleProvider(api_base="https://mygateway.example/v1", api_key="explicit-key")
    assert provider.client.api_key == "explicit-key"


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "env-key")
    provider = OpenAICompatibleProvider(api_base="https://mygateway.example/v1")
    assert provider.client.api_key == "env-key"


def test_explicit_api_key_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "env-key")
    provider = OpenAICompatibleProvider(api_base="https://mygateway.example/v1", api_key="explicit-key")
    assert provider.client.api_key == "explicit-key"


def test_allow_env_api_key_defaults_to_reading_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "env-key")
    provider = AnyLLM.create_openai_compatible(name="mygateway", api_base="https://mygateway.example/v1")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.client.api_key == "env-key"


def test_allow_env_api_key_false_ignores_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A gateway serving endpoints it did not choose must never forward its own
    # deployment-level key to one of them.
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "env-key")
    provider = AnyLLM.create_openai_compatible(
        name="mygateway", api_base="https://mygateway.example/v1", allow_env_api_key=False
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.client.api_key == "no-key-required"


def test_allow_env_api_key_false_still_uses_explicit_key() -> None:
    provider = AnyLLM.create_openai_compatible(
        name="mygateway", api_base="https://mygateway.example/v1", api_key="explicit-key", allow_env_api_key=False
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.client.api_key == "explicit-key"


def test_allow_env_api_key_false_keeps_custom_identity() -> None:
    provider = AnyLLM.create_openai_compatible(
        name="mygateway", api_base="https://mygateway.example/v1", allow_env_api_key=False
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.PROVIDER_NAME == "mygateway"
    assert provider.get_provider_metadata().name == "mygateway"


def test_capability_flags_follow_openai_compatible_defaults() -> None:
    provider = AnyLLM.create_openai_compatible(name="mygateway", api_base="https://mygateway.example/v1")
    assert provider.PROMPT_CACHE_KEY_SUPPORT == "passthrough"
    assert provider.SUPPORTS_COMPLETION
    assert provider.SUPPORTS_COMPLETION_STREAMING
    assert provider.SUPPORTS_LIST_MODELS
    assert not provider.SUPPORTS_RESPONSES
    assert not provider.SUPPORTS_BATCH


def test_client_kwargs_are_forwarded() -> None:
    provider = AnyLLM.create_openai_compatible(
        name="mygateway", api_base="https://mygateway.example/v1", api_key="k", timeout=12.5
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.client.timeout == 12.5


def test_custom_name_is_not_an_enum_member() -> None:
    # The custom path deliberately lives outside the provider enum: a custom endpoint is
    # represented by an instance, not a registered provider key.
    with pytest.raises(UnsupportedProviderError):
        LLMProvider.from_string("mygateway")
