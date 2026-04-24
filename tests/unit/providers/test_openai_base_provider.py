import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openresponses_types import ResponseResource

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import CompletionParams
from any_llm.types.request import RequestParams
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


def test_base_provider_converts_dataclass_response_format_to_json_schema() -> None:
    """Test that plain dataclasses are converted to JSON schema dicts."""
    from dataclasses import dataclass

    @dataclass
    class TestOutput:
        name: str
        value: int

    params = CompletionParams(
        model_id="model",
        messages=[{"role": "user", "content": "hi"}],
        response_format=TestOutput,
    )
    result = BaseOpenAIProvider._convert_completion_params(params)

    assert result["response_format"]["type"] == "json_schema"
    assert result["response_format"]["json_schema"]["name"] == "TestOutput"
    assert "properties" in result["response_format"]["json_schema"]["schema"]


@pytest.mark.asyncio
async def test_acompletion_with_dataclass_uses_create_not_parse() -> None:
    """Test that plain dataclasses use .create() instead of .parse()."""
    from dataclasses import dataclass

    @dataclass
    class TestOutput:
        name: str

    class TestProvider(BaseOpenAIProvider):
        PROVIDER_NAME = "TestProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai_class:
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        provider = TestProvider(api_key="test-key")
        await provider._acompletion(
            CompletionParams(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=TestOutput,
            )
        )

        mock_client.chat.completions.create.assert_called_once()
        mock_client.chat.completions.parse.assert_not_called()


@pytest.mark.asyncio
async def test_arequest_delegates_to_aresponses() -> None:
    class TestProvider(BaseOpenAIProvider):
        PROVIDER_NAME = "TestProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"

    response = ResponseResource.model_validate(
        {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "completed_at": 0,
            "status": "completed",
            "incomplete_details": None,
            "model": "gpt-test",
            "previous_response_id": None,
            "instructions": None,
            "output": [],
            "error": None,
            "tools": [],
            "tool_choice": "auto",
            "truncation": "disabled",
            "parallel_tool_calls": False,
            "text": {"format": {"type": "text"}},
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "top_logprobs": 0,
            "temperature": 1.0,
            "reasoning": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "max_output_tokens": None,
            "max_tool_calls": None,
            "store": False,
            "background": False,
            "service_tier": "default",
            "metadata": {},
            "safety_identifier": None,
            "prompt_cache_key": None,
        }
    )

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = TestProvider(api_key="test-key")
        provider._aresponses = AsyncMock(return_value=response)  # type: ignore[method-assign]

        result = await provider._arequest(RequestParams(model="gpt-test", input="Hello"))

        assert result == response
        provider._aresponses.assert_called_once()
