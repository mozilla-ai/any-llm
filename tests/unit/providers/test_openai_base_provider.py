from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from openai.types.responses import Response as OpenAIResponse
from openresponses_types import CreateResponseBody, ResponseResource

from any_llm.providers.openai.base import BaseOpenAIProvider
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


class ResponsesProvider(BaseOpenAIProvider):
    """Test provider subclass with responses support enabled."""

    SUPPORTS_RESPONSES = True
    PROVIDER_NAME = "ResponsesProvider"
    ENV_API_KEY_NAME = "TEST_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://example.com"
    API_BASE = "https://api.example.com/v1"


def _create_mock_openai_response() -> dict[str, Any]:
    """Create a mock OpenAI Response object dict with all required fields."""
    return {
        "id": "resp_123abc",
        "object": "response",
        "created_at": 1700000000,
        "completed_at": 1700000005,
        "status": "completed",
        "model": "gpt-4o-2024-08-06",
        "instructions": "You are a helpful assistant.",
        "temperature": 0.7,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "top_logprobs": 0,
        "max_output_tokens": 1000,
        "max_tool_calls": None,
        "parallel_tool_calls": True,
        "truncation": "auto",
        "store": False,
        "background": False,
        "metadata": {"request_id": "req_456"},
        "output": [
            {
                "type": "message",
                "id": "msg_001",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello! How can I help you today?",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ],
        "tools": [],
        "tool_choice": "auto",
        "text": {"format": {"type": "text"}, "verbosity": None},
        "reasoning": {
            "effort": "medium",
            "summary": None,
        },
        "usage": {
            "input_tokens": 10,
            "output_tokens": 15,
            "total_tokens": 25,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "error": None,
        "incomplete_details": None,
        "previous_response_id": None,
        "service_tier": "default",
        "safety_identifier": None,
        "prompt_cache_key": None,
    }


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_converts_openai_response_to_response_resource(mock_openai_class: MagicMock) -> None:
    """Test that OpenAI Response is correctly converted to ResponseResource."""
    mock_response_data = _create_mock_openai_response()

    mock_response = Mock(spec=OpenAIResponse)
    mock_response.model_dump.return_value = mock_response_data

    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=mock_response)
    mock_openai_class.return_value = mock_client

    provider = ResponsesProvider(api_key="test-key")

    params = CreateResponseBody(
        model="gpt-4o-2024-08-06",
        input=[{"type": "message", "role": "user", "content": "Hello"}],  # type: ignore[list-item]
    )
    result = await provider._aresponses(params)

    assert isinstance(result, ResponseResource)
    assert result.id == "resp_123abc"
    assert result.object == "response"
    assert result.created_at == 1700000000
    assert result.completed_at == 1700000005
    assert result.status == "completed"
    assert result.model == "gpt-4o-2024-08-06"
    assert result.instructions == "You are a helpful assistant."
    assert result.temperature == 0.7
    assert result.metadata == {"request_id": "req_456"}

    mock_client.responses.create.assert_called_once()


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_preserves_output_content(mock_openai_class: MagicMock) -> None:
    """Test that output items are correctly preserved during conversion."""
    mock_response_data = _create_mock_openai_response()

    mock_response = Mock(spec=OpenAIResponse)
    mock_response.model_dump.return_value = mock_response_data

    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=mock_response)
    mock_openai_class.return_value = mock_client

    provider = ResponsesProvider(api_key="test-key")

    params = CreateResponseBody(
        model="gpt-4o-2024-08-06",
        input=[{"type": "message", "role": "user", "content": "Hello"}],  # type: ignore[list-item]
    )
    result = await provider._aresponses(params)

    assert isinstance(result, ResponseResource)
    assert result.output is not None
    assert len(result.output) == 1
    output_item = result.output[0]
    assert output_item.type == "message"
    assert output_item.role == "assistant"


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_preserves_reasoning(mock_openai_class: MagicMock) -> None:
    """Test that reasoning content is preserved during conversion."""
    mock_response_data = _create_mock_openai_response()

    mock_response = Mock(spec=OpenAIResponse)
    mock_response.model_dump.return_value = mock_response_data

    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=mock_response)
    mock_openai_class.return_value = mock_client

    provider = ResponsesProvider(api_key="test-key")

    params = CreateResponseBody(
        model="gpt-4o-2024-08-06",
        input=[{"type": "message", "role": "user", "content": "Hello"}],  # type: ignore[list-item]
    )
    result = await provider._aresponses(params)

    assert isinstance(result, ResponseResource)
    assert result.reasoning is not None


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_preserves_usage_stats(mock_openai_class: MagicMock) -> None:
    """Test that usage statistics are preserved during conversion."""
    mock_response_data = _create_mock_openai_response()

    mock_response = Mock(spec=OpenAIResponse)
    mock_response.model_dump.return_value = mock_response_data

    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=mock_response)
    mock_openai_class.return_value = mock_client

    provider = ResponsesProvider(api_key="test-key")

    params = CreateResponseBody(
        model="gpt-4o-2024-08-06",
        input=[{"type": "message", "role": "user", "content": "Hello"}],  # type: ignore[list-item]
    )
    result = await provider._aresponses(params)

    assert isinstance(result, ResponseResource)
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 15
    assert result.usage.total_tokens == 25


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_handles_response_with_null_optional_fields(mock_openai_class: MagicMock) -> None:
    """Test conversion handles response where optional fields are null."""
    response_data = {
        "id": "resp_minimal",
        "object": "response",
        "created_at": 1700000000,
        "completed_at": None,
        "status": "completed",
        "model": "gpt-4o",
        "instructions": None,
        "temperature": 1.0,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "top_logprobs": 0,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "parallel_tool_calls": True,
        "truncation": "auto",
        "store": False,
        "background": False,
        "metadata": {},
        "output": [],
        "tools": [],
        "tool_choice": "auto",
        "text": {"format": {"type": "text"}, "verbosity": None},
        "reasoning": None,
        "usage": None,
        "error": None,
        "incomplete_details": None,
        "previous_response_id": None,
        "service_tier": "default",
        "safety_identifier": None,
        "prompt_cache_key": None,
    }

    mock_response = Mock(spec=OpenAIResponse)
    mock_response.model_dump.return_value = response_data

    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=mock_response)
    mock_openai_class.return_value = mock_client

    provider = ResponsesProvider(api_key="test-key")

    params = CreateResponseBody(
        model="gpt-4o",
        input=[{"type": "message", "role": "user", "content": "Hello"}],  # type: ignore[list-item]
    )
    result = await provider._aresponses(params)

    assert isinstance(result, ResponseResource)
    assert result.id == "resp_minimal"
    assert result.model == "gpt-4o"
    assert result.output == []
    assert result.reasoning is None
    assert result.usage is None
    assert result.error is None


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_streaming_returns_iterator(mock_openai_class: MagicMock) -> None:
    """Test that streaming responses return an async iterator of dicts."""
    from openai._streaming import AsyncStream

    mock_event_1 = Mock()
    mock_event_1.model_dump.return_value = {"type": "response.created", "response": {"id": "resp_123"}}

    mock_event_2 = Mock()
    mock_event_2.model_dump.return_value = {"type": "response.done", "response": {"id": "resp_123"}}

    async def async_generator() -> AsyncIterator[Mock]:
        yield mock_event_1
        yield mock_event_2

    mock_stream = Mock(spec=AsyncStream)
    mock_stream.__aiter__ = lambda self: async_generator()

    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=mock_stream)
    mock_openai_class.return_value = mock_client

    provider = ResponsesProvider(api_key="test-key")

    params = CreateResponseBody(
        model="gpt-4o",
        input=[{"type": "message", "role": "user", "content": "Hello"}],  # type: ignore[list-item]
        stream=True,
    )
    result = await provider._aresponses(params)

    events = []
    async for event in result:  # type: ignore[union-attr]
        events.append(event)

    assert len(events) == 2
    assert events[0]["type"] == "response.created"
    assert events[1]["type"] == "response.done"
