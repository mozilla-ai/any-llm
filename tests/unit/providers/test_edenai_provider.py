from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.edenai import EdenaiProvider
from any_llm.providers.edenai.utils import _convert_models_list
from any_llm.types.completion import CompletionParams
from any_llm.types.model import Model


def test_edenai_provider_attributes() -> None:
    """Eden AI provider declares the expected config and capability flags."""
    assert EdenaiProvider.PROVIDER_NAME == "edenai"
    assert EdenaiProvider.API_BASE == "https://api.edenai.run/v3"
    assert EdenaiProvider.ENV_API_KEY_NAME == "EDENAI_API_KEY"
    assert EdenaiProvider.SUPPORTS_COMPLETION is True
    assert EdenaiProvider.SUPPORTS_RESPONSES is False


def test_edenai_provider_raises_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiating without a key (and no env var) raises MissingApiKeyError."""
    monkeypatch.delenv("EDENAI_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        EdenaiProvider()


def test_edenai_remaps_max_tokens_to_max_completion_tokens() -> None:
    """Base OpenAI-compatible behavior: max_tokens is remapped for the chat API."""
    params = CompletionParams(
        model_id="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=8192,
    )
    result = EdenaiProvider._convert_completion_params(params)
    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 8192


def _make_edenai_model(**overrides: object) -> Model:
    """Build a fake model resembling what the OpenAI SDK constructs from Eden AI's response."""
    defaults: dict[str, object] = {
        "id": "openai/gpt-4o-mini",
        "created": None,
        "object": None,
        "owned_by": None,
    }
    defaults.update(overrides)
    return Model.model_construct(**defaults)  # type: ignore[arg-type]


def test_convert_models_list_fills_missing_fields() -> None:
    """Models with None object/created are rebuilt with valid defaults."""
    response = SimpleNamespace(data=[_make_edenai_model(created=1779376861)])
    result = _convert_models_list(response)

    assert len(result) == 1
    model = result[0]
    assert model.id == "openai/gpt-4o-mini"
    assert model.object == "model"
    assert model.created == 1779376861


def test_convert_models_list_owned_by_from_vendor_prefix() -> None:
    """owned_by is derived from the vendor prefix of Eden AI's id."""
    response = SimpleNamespace(data=[_make_edenai_model(id="anthropic/claude-sonnet-4-5")])
    result = _convert_models_list(response)

    assert result[0].owned_by == "anthropic"


def test_convert_models_list_owned_by_fallback_when_no_prefix() -> None:
    """An id without a vendor prefix falls back to 'edenai'."""
    response = SimpleNamespace(data=[_make_edenai_model(id="some-model")])
    result = _convert_models_list(response)

    assert result[0].owned_by == "edenai"


def test_convert_models_list_defaults_created_to_zero_when_none() -> None:
    """A None created timestamp falls back to 0."""
    response = SimpleNamespace(data=[_make_edenai_model(created=None)])
    result = _convert_models_list(response)

    assert result[0].created == 0


def test_convert_models_list_handles_empty_response() -> None:
    """An empty response returns an empty list."""
    response = SimpleNamespace(data=[])
    result = _convert_models_list(response)

    assert result == []


def test_convert_models_list_round_trip_serialization() -> None:
    """Converted models must survive model_dump_json -> model_validate_json."""
    response = SimpleNamespace(data=[_make_edenai_model()])
    models = _convert_models_list(response)

    for model in models:
        json_data = model.model_dump_json()
        restored = Model.model_validate_json(json_data)
        assert restored.id == model.id
        assert restored.object == "model"
        assert restored.owned_by == "openai"


def test_convert_models_list_preserves_extra_edenai_fields() -> None:
    """Extra Eden AI attributes (model_name, context_length) survive conversion."""
    model = _make_edenai_model(model_name="gpt-4o-mini", context_length=128000)
    response = SimpleNamespace(data=[model])
    result = _convert_models_list(response)

    converted = result[0]
    assert converted.model_extra is not None
    assert converted.model_extra["model_name"] == "gpt-4o-mini"
    assert converted.model_extra["context_length"] == 128000


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_list_models_returns_valid_model_objects(mock_openai_class: MagicMock) -> None:
    """End-to-end: EdenaiProvider.list_models() returns spec-compliant Model objects."""
    raw_models = [
        _make_edenai_model(id="openai/gpt-4o-mini"),
        _make_edenai_model(id="mistral/mistral-small-latest"),
    ]
    mock_client = AsyncMock()
    mock_client.models.list.return_value = SimpleNamespace(data=raw_models)
    mock_openai_class.return_value = mock_client

    provider = EdenaiProvider(api_key="sk-test")
    result = provider.list_models()

    assert len(result) == 2
    for model in result:
        assert isinstance(model, Model)
        assert model.object == "model"
        assert model.owned_by is not None
    assert result[0].owned_by == "openai"
    assert result[1].owned_by == "mistral"
