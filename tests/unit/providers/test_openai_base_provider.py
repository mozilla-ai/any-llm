import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.openai.openai import OpenaiProvider
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


@pytest.mark.asyncio
async def test_stream_with_response_format_raises() -> None:
    class TestProvider(BaseOpenAIProvider):
        PROVIDER_NAME = "TestProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = TestProvider(api_key="test-key")

        with pytest.raises(ValueError, match="stream is not supported for response_format"):
            await provider._acompletion(
                CompletionParams(
                    model_id="test-model",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True,
                    response_format={"type": "json_object"},
                )
            )


def test_base_provider_maps_max_tokens_to_max_completion_tokens() -> None:
    params = CompletionParams(model_id="model", messages=[{"role": "user", "content": "hi"}], max_tokens=8192)
    result = BaseOpenAIProvider._convert_completion_params(params)
    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 8192


def test_base_provider_preserves_explicit_max_completion_tokens() -> None:
    params = CompletionParams(
        model_id="model",
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=4096,
    )
    result = BaseOpenAIProvider._convert_completion_params(params)
    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 4096


def test_base_provider_max_completion_tokens_takes_precedence_over_max_tokens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    params = CompletionParams(
        model_id="model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8192,
        max_completion_tokens=4096,
    )

    any_llm_logger = logging.getLogger("any_llm")
    any_llm_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="any_llm"):
            result = BaseOpenAIProvider._convert_completion_params(params)
    finally:
        any_llm_logger.propagate = False

    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 4096
    assert "Ignoring max_tokens (8192) in favor of max_completion_tokens (4096)" in caplog.text


def test_base_provider_no_max_tokens_passes_through_unchanged() -> None:
    params = CompletionParams(model_id="model", messages=[{"role": "user", "content": "hi"}], temperature=0.5)
    result = BaseOpenAIProvider._convert_completion_params(params)
    assert "max_tokens" not in result
    assert "max_completion_tokens" not in result
    assert result["temperature"] == 0.5


def test_base_provider_max_tokens_via_kwargs_also_remapped() -> None:
    params = CompletionParams(model_id="model", messages=[{"role": "user", "content": "hi"}])
    result = BaseOpenAIProvider._convert_completion_params(params, max_tokens=1024)
    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 1024


def test_base_openai_provider_excludes_prompt_cache_fields() -> None:
    params = CompletionParams(
        model_id="model",
        messages=[{"role": "user", "content": "hi"}],
        prompt_cache_key="my-key",
        prompt_cache_retention="1h",
    )
    result = BaseOpenAIProvider._convert_completion_params(params)
    assert "prompt_cache_key" not in result
    assert "prompt_cache_retention" not in result


def test_openai_provider_includes_prompt_cache_fields_when_set() -> None:
    params = CompletionParams(
        model_id="gpt-5.2",
        messages=[{"role": "user", "content": "hi"}],
        prompt_cache_key="my-key",
        prompt_cache_retention="1h",
    )
    result = OpenaiProvider._convert_completion_params(params)
    assert result["prompt_cache_key"] == "my-key"
    assert result["prompt_cache_retention"] == "1h"


def test_openai_provider_omits_prompt_cache_fields_when_none() -> None:
    params = CompletionParams(
        model_id="gpt-5.2",
        messages=[{"role": "user", "content": "hi"}],
    )
    result = OpenaiProvider._convert_completion_params(params)
    assert "prompt_cache_key" not in result
    assert "prompt_cache_retention" not in result
