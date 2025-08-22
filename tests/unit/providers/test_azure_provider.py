from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pytest

from any_llm.exceptions import UnsupportedModelResponseError
from any_llm.provider import ApiConfig
from any_llm.providers.azure.azure import AzureProvider
from any_llm.types.completion import CompletionParams
from any_llm.types.model import Model


@contextmanager
def mock_azure_provider() -> Any:
    with (
        patch("any_llm.providers.azure.azure.aio.ChatCompletionsClient") as mock_chat_client,
        patch("any_llm.providers.azure.azure._convert_response") as mock_convert_response,
    ):
        mock_client_instance = MagicMock()
        mock_chat_client.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_client_instance.complete = AsyncMock(return_value=mock_response)

        yield mock_client_instance, mock_convert_response, mock_chat_client


@contextmanager
def mock_azure_streaming_provider() -> Any:
    with (
        patch("any_llm.providers.azure.azure.aio.ChatCompletionsClient") as mock_chat_client,
        patch("any_llm.providers.azure.azure._stream_completion_async") as mock_stream_completion,
    ):
        mock_client_instance = MagicMock()
        mock_chat_client.return_value = mock_client_instance

        mock_openai_chunk1 = MagicMock()
        mock_openai_chunk2 = MagicMock()
        mock_stream_completion.return_value = [mock_openai_chunk1, mock_openai_chunk2]

        yield mock_client_instance, mock_stream_completion, mock_chat_client


@pytest.mark.asyncio
async def test_azure_with_api_key_and_api_base() -> None:
    api_key = "test-api-key"
    custom_endpoint = "https://test.eu.models.ai.azure.com"

    messages = [{"role": "user", "content": "Hello"}]
    with mock_azure_provider() as (mock_client, mock_convert_response, mock_chat_client):
        provider = AzureProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))
        await provider.acompletion(CompletionParams(model_id="model-id", messages=messages))

        mock_chat_client.assert_called_once()

        mock_client.complete.assert_called_once_with(
            model="model-id",
            messages=messages,
        )

        mock_convert_response.assert_called_once_with(mock_client.complete.return_value)


@pytest.mark.asyncio
async def test_azure_with_tools() -> None:
    api_key = "test-api-key"
    custom_endpoint = "https://aoairesource.openai.azure.com"

    messages = [{"role": "user", "content": "Hello"}]
    tools = {"type": "function", "function": "foo"}
    tool_choice = "auto"
    with mock_azure_provider() as (mock_client, mock_convert_response, mock_chat_client):
        provider = AzureProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))
        await provider.acompletion(
            CompletionParams(
                model_id="model-id",
                messages=messages,
                tools=[tools] if isinstance(tools, dict) else tools,
                tool_choice=tool_choice,
            )
        )

        mock_client.complete.assert_called_once_with(
            model="model-id",
            messages=messages,
            tools=[tools],
            tool_choice=tool_choice,
        )

        mock_convert_response.assert_called_once_with(mock_client.complete.return_value)


@pytest.mark.asyncio
async def test_azure_streaming() -> None:
    api_key = "test-api-key"
    custom_endpoint = "https://test.eu.models.ai.azure.com"

    messages = [{"role": "user", "content": "Hello"}]

    provider = AzureProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))

    with patch.object(provider, "_stream_completion_async") as mock_stream_completion:
        mock_openai_chunk1 = MagicMock()
        mock_openai_chunk2 = MagicMock()
        mock_stream_completion.return_value = [mock_openai_chunk1, mock_openai_chunk2]

        result = await provider.acompletion(CompletionParams(model_id="model-id", messages=messages, stream=True))

        assert mock_stream_completion.call_count == 1

        call_args = mock_stream_completion.call_args
        assert call_args is not None
        args, kwargs = call_args
        assert len(args) >= 3  # client, model, messages
        assert args[1] == "model-id"  # model
        assert args[2] == messages  # messages
        assert kwargs.get("stream") is True

        assert isinstance(result, list)
        assert len(result) == 2


def test_azure_models_raises_unsupported_model_response_error() -> None:
    """Test that list_models raises UnsupportedModelResponseError on client failure."""
    api_key = "test-api-key"
    custom_endpoint = "https://test.eu.models.ai.azure.com"
    provider = AzureProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))

    with patch("any_llm.providers.azure.azure.AIProjectClient") as mock_project_client:
        mock_project_client.side_effect = Exception("mocked error")
        with pytest.raises(UnsupportedModelResponseError) as exc_info:
            provider.list_models()
        exc = exc_info.value
        assert "Failed to parse Azure model response." in str(exc)


def test_azure_models_returns_model_metadata(monkeypatch: Any) -> None:
    api_key = "test-api-key"
    api_version = "2024-08-01-preview"
    custom_endpoint = "https://test.eu.models.ai.azure.com"
    provider = AzureProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))

    with patch("any_llm.providers.azure.azure.AIProjectClient") as mock_project_client:
        mock_model_data = [
            Model(id="gpt-3.5-turbo", object="model", created=1677610602, owned_by="openai"),
            Model(id="gpt-4", object="model", created=1687882411, owned_by="openai"),
        ]

        mock_project_instance = MagicMock()
        mock_project_client.return_value = mock_project_instance
        mock_project_instance.get_openai_client(api_version=api_version).models.list.return_value = mock_model_data

        result = provider.list_models()

        for actual, expected in zip(result, mock_model_data, strict=False):
            assert actual.id == expected.id
            assert actual.object == expected.object
            assert actual.created == expected.created
            assert actual.owned_by == expected.owned_by
