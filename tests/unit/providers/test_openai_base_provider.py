from unittest.mock import MagicMock, patch

import pytest

from any_llm.exceptions import UnsupportedModelResponseError
from any_llm.provider import ApiConfig
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.model import Model


@patch("any_llm.providers.openai.base.OpenAI")
def test_list_models_returns_model_list_when_supported(mock_openai_class: MagicMock) -> None:
    class ListModelsProvider(BaseOpenAIProvider):
        SUPPORTS_LIST_MODELS = True
        PROVIDER_NAME = "ListModelsProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"
        API_BASE = "https://api.example.com/v1"

    mock_model_data = [
        Model(id="gpt-3.5-turbo", object="model", created=1677610602, owned_by="openai"),
        Model(id="gpt-4", object="model", created=1687882411, owned_by="openai"),
    ]

    mock_client = MagicMock()
    mock_client.models.list.return_value = mock_model_data
    mock_openai_class.return_value = mock_client

    config = ApiConfig(api_key="test-key", api_base="https://custom.api.com/v1")
    provider = ListModelsProvider(config=config)

    result = provider.list_models()

    for actual, expected in zip(result, mock_model_data, strict=False):
        assert actual.id == expected.id
        assert actual.object == expected.object
        assert actual.created == expected.created
        assert actual.owned_by == expected.owned_by
    mock_openai_class.assert_called_once_with(base_url="https://custom.api.com/v1", api_key="test-key")
    mock_client.models.list.assert_called_once_with()


@patch("any_llm.providers.openai.base.OpenAI")
def test_list_models_uses_default_api_base_when_not_configured(mock_openai_class: MagicMock) -> None:
    class ListModelsProvider(BaseOpenAIProvider):
        SUPPORTS_LIST_MODELS = True
        PROVIDER_NAME = "ListModelsProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"
        API_BASE = "https://api.default.com/v1"

    mock_client = MagicMock()
    mock_client.models.list.return_value.data = []
    mock_openai_class.return_value = mock_client

    config = ApiConfig(api_key="test-key")
    provider = ListModelsProvider(config=config)

    provider.list_models()

    mock_openai_class.assert_called_once_with(base_url="https://api.default.com/v1", api_key="test-key")


@patch("any_llm.providers.openai.base.OpenAI")
def test_list_models_passes_kwargs_to_client(mock_openai_class: MagicMock) -> None:
    class ListModelsProvider(BaseOpenAIProvider):
        SUPPORTS_LIST_MODELS = True
        PROVIDER_NAME = "ListModelsProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"

    mock_client = MagicMock()
    mock_client.models.list.return_value.data = []
    mock_openai_class.return_value = mock_client

    config = ApiConfig(api_key="test-key")
    provider = ListModelsProvider(config=config)

    provider.list_models(limit=10, after="model-123")

    mock_client.models.list.assert_called_once_with(limit=10, after="model-123")


def test_models_raises_not_implemented_if_not_supported() -> None:
    class NoListModelsProvider(BaseOpenAIProvider):
        SUPPORTS_LIST_MODELS = False
        PROVIDER_NAME = "NoListModelsProvider"

    config = ApiConfig(api_key="test-key")
    provider = NoListModelsProvider(config=config)
    with pytest.raises(NotImplementedError):
        provider.list_models()


@patch("any_llm.providers.openai.base.OpenAI")
def test_models_raises_unsupported_model_response_error(mock_openai: MagicMock) -> None:
    class ListModelsProvider(BaseOpenAIProvider):
        SUPPORTS_LIST_MODELS = True
        PROVIDER_NAME = "ListModelsProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"
        API_BASE = "https://api.default.com/v1"

    config = ApiConfig(api_key="test-key")
    provider = ListModelsProvider(config=config)
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.models.list.side_effect = RuntimeError("bad response")

    with pytest.raises(UnsupportedModelResponseError) as excinfo:
        provider.list_models()
    assert "Failed to parse OpenAI model response." in str(excinfo.value)
    assert isinstance(excinfo.value.original_exception, RuntimeError)
