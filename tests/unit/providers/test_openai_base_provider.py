import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import CompletionParams
from any_llm.types.model import Model


@patch("any_llm.providers.openai.base.AsyncOpenAI")
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

    mock_client = AsyncMock()
    mock_client.models.list.return_value.data = mock_model_data
    mock_openai_class.return_value = mock_client

    provider = ListModelsProvider(api_key="test-key", api_base="https://custom.api.com/v1")

    result = provider.list_models()

    assert result == mock_model_data
    mock_openai_class.assert_called_once_with(base_url="https://custom.api.com/v1", api_key="test-key")
    mock_client.models.list.assert_called_once_with()


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_list_models_uses_default_api_base_when_not_configured(mock_openai_class: MagicMock) -> None:
    class ListModelsProvider(BaseOpenAIProvider):
        SUPPORTS_LIST_MODELS = True
        PROVIDER_NAME = "ListModelsProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"
        API_BASE = "https://api.default.com/v1"

    mock_client = AsyncMock()
    mock_client.models.list.return_value.data = []
    mock_openai_class.return_value = mock_client

    provider = ListModelsProvider(api_key="test-key")

    provider.list_models()

    mock_openai_class.assert_called_once_with(base_url="https://api.default.com/v1", api_key="test-key")


@patch(
    "any_llm.providers.openai.base.AsyncOpenAI",
)
def test_list_models_passes_kwargs_to_client(mock_openai_class: MagicMock) -> None:
    class ListModelsProvider(BaseOpenAIProvider):
        SUPPORTS_LIST_MODELS = True
        PROVIDER_NAME = "ListModelsProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"

    mock_client = AsyncMock()
    mock_client.models.list.return_value.data = []
    mock_openai_class.return_value = mock_client

    provider = ListModelsProvider(api_key="test-key")

    provider.list_models(limit=10, after="model-123")

    mock_client.models.list.assert_called_once_with(limit=10, after="model-123")


def test_convert_completion_params_maps_max_tokens_to_max_completion_tokens() -> None:
    """Test that max_tokens is mapped to max_completion_tokens for the OpenAI API."""
    params = CompletionParams(
        model_id="gpt-5.2",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=8192,
    )

    result = BaseOpenAIProvider._convert_completion_params(params)

    assert "max_completion_tokens" in result
    assert result["max_completion_tokens"] == 8192
    assert "max_tokens" not in result


def test_convert_completion_params_preserves_explicit_max_completion_tokens(caplog: pytest.LogCaptureFixture) -> None:
    """Test that explicit max_completion_tokens is not overwritten by max_tokens."""
    params = CompletionParams(
        model_id="gpt-5.2",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=4096,
        max_completion_tokens=8192,
    )

    any_llm_logger = logging.getLogger("any_llm")
    any_llm_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="any_llm"):
            result = BaseOpenAIProvider._convert_completion_params(params)
    finally:
        any_llm_logger.propagate = False

    assert result["max_completion_tokens"] == 8192
    assert "max_tokens" not in result
    assert "Ignoring max_tokens (4096) in favor of max_completion_tokens (8192)" in caplog.text


def test_convert_completion_params_without_max_tokens() -> None:
    """Test that params without max_tokens pass through unchanged."""
    params = CompletionParams(
        model_id="gpt-5.2",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
    )

    result = BaseOpenAIProvider._convert_completion_params(params)

    assert "max_tokens" not in result
    assert "max_completion_tokens" not in result
    assert result["temperature"] == 0.7


def test_convert_completion_params_max_completion_tokens_only() -> None:
    """Test that max_completion_tokens alone passes through correctly."""
    params = CompletionParams(
        model_id="gpt-5.2",
        messages=[{"role": "user", "content": "Hello"}],
        max_completion_tokens=1024,
    )

    result = BaseOpenAIProvider._convert_completion_params(params)

    assert result["max_completion_tokens"] == 1024
    assert "max_tokens" not in result


def test_convert_completion_params_kwargs_max_tokens_remapped() -> None:
    """Test that max_tokens passed via kwargs is remapped to max_completion_tokens."""
    params = CompletionParams(
        model_id="gpt-5.2",
        messages=[{"role": "user", "content": "Hello"}],
    )

    result = BaseOpenAIProvider._convert_completion_params(params, max_tokens=2048)

    assert result["max_completion_tokens"] == 2048
    assert "max_tokens" not in result


def test_convert_completion_params_kwargs_override_other_params() -> None:
    """Test that kwargs override params values for non-max_tokens fields."""
    params = CompletionParams(
        model_id="gpt-5.2",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
    )

    result = BaseOpenAIProvider._convert_completion_params(params, temperature=0.2)

    assert result["temperature"] == 0.2


def test_convert_completion_params_conflict_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test that providing both max_tokens and max_completion_tokens via kwargs logs a warning."""
    params = CompletionParams(
        model_id="gpt-5.2",
        messages=[{"role": "user", "content": "Hello"}],
    )

    any_llm_logger = logging.getLogger("any_llm")
    any_llm_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="any_llm"):
            result = BaseOpenAIProvider._convert_completion_params(params, max_tokens=4096, max_completion_tokens=8192)
    finally:
        any_llm_logger.propagate = False

    assert result["max_completion_tokens"] == 8192
    assert "max_tokens" not in result
    assert "Ignoring max_tokens (4096) in favor of max_completion_tokens (8192)" in caplog.text


def test_convert_completion_params_kwargs_max_completion_tokens_wins_over_params_max_tokens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that max_completion_tokens via kwargs wins over max_tokens in params."""
    params = CompletionParams(
        model_id="gpt-5.2",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=4096,
    )

    any_llm_logger = logging.getLogger("any_llm")
    any_llm_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="any_llm"):
            result = BaseOpenAIProvider._convert_completion_params(params, max_completion_tokens=8192)
    finally:
        any_llm_logger.propagate = False

    assert result["max_completion_tokens"] == 8192
    assert "max_tokens" not in result
    assert "Ignoring max_tokens (4096) in favor of max_completion_tokens (8192)" in caplog.text
