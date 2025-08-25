from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.provider import ApiConfig
from any_llm.providers.azure.azure import AzureProvider
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_azure_provider():  # type: ignore[no-untyped-def]
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
def mock_azure_streaming_provider():  # type: ignore[no-untyped-def]
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
async def test_azure_with_api_version() -> None:
    api_key = "test-api-key"
    custom_endpoint = "https://test.eu.models.ai.azure.com"

    messages = [{"role": "user", "content": "Hello"}]
    with mock_azure_provider() as (_, _, mock_chat_client):
        with patch("any_llm.providers.azure.azure.AzureKeyCredential") as mock_azure_key_credential:
            provider = AzureProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))
            await provider.acompletion(
                CompletionParams(model_id="model-id", messages=messages), api_version="2025-04-01-preview"
            )

            mock_chat_client.assert_called_once_with(
                endpoint=provider._get_endpoint(),
                credential=mock_azure_key_credential(api_key),
                api_version="2025-04-01-preview",
            )


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
