from contextlib import contextmanager
from unittest.mock import patch, MagicMock

from any_llm.provider import ApiConfig
from any_llm.providers.azure.azure import AzureProvider


@contextmanager
def mock_azure_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.azure.azure.ChatCompletionsClient") as mock_chat_client,
        patch("any_llm.providers.azure.azure._convert_response") as mock_convert_response,
    ):
        mock_client_instance = MagicMock()
        mock_chat_client.return_value = mock_client_instance

        # Mock the complete method
        mock_response = MagicMock()
        mock_client_instance.complete.return_value = mock_response

        yield mock_client_instance, mock_convert_response, mock_chat_client


@contextmanager
def mock_azure_streaming_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.azure.azure.ChatCompletionsClient") as mock_chat_client,
        patch("any_llm.providers.azure.azure._stream_completion") as mock_stream_completion,
    ):
        mock_client_instance = MagicMock()
        mock_chat_client.return_value = mock_client_instance

        # Mock the streaming response
        mock_openai_chunk1 = MagicMock()
        mock_openai_chunk2 = MagicMock()
        mock_stream_completion.return_value = [mock_openai_chunk1, mock_openai_chunk2]

        yield mock_client_instance, mock_stream_completion, mock_chat_client


def test_azure_with_api_key_and_api_base() -> None:
    api_key = "test-api-key"
    custom_endpoint = "https://test.eu.models.ai.azure.com"

    messages = [{"role": "user", "content": "Hello"}]
    with mock_azure_provider() as (mock_client, mock_convert_response, mock_chat_client):
        provider = AzureProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))
        provider._make_api_call("model-id", messages)

        # Verify ChatCompletionsClient was created with correct parameters
        mock_chat_client.assert_called_once()

        # Verify the complete method was called with correct parameters
        mock_client.complete.assert_called_once_with(
            model="model-id",
            messages=messages,
        )

        # Verify response conversion was called
        mock_convert_response.assert_called_once_with(mock_client.complete.return_value)


def test_azure_with_tools() -> None:
    api_key = "test-api-key"
    custom_endpoint = "https://aoairesource.openai.azure.com"

    messages = [{"role": "user", "content": "Hello"}]
    tools = {"type": "function", "function": "foo"}
    tool_choice = "auto"
    with mock_azure_provider() as (mock_client, mock_convert_response, mock_chat_client):
        provider = AzureProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))
        provider._make_api_call("model-id", messages, tools=tools, tool_choice=tool_choice)

        # Verify the complete method was called with correct parameters including tools
        mock_client.complete.assert_called_once_with(
            model="model-id",
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

        # Verify response conversion was called
        mock_convert_response.assert_called_once_with(mock_client.complete.return_value)


def test_azure_streaming() -> None:
    api_key = "test-api-key"
    custom_endpoint = "https://test.eu.models.ai.azure.com"

    messages = [{"role": "user", "content": "Hello"}]

    # Create the provider first
    provider = AzureProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))

    # Mock the _stream_completion method on the provider instance
    with patch.object(provider, "_stream_completion") as mock_stream_completion:
        # Mock the streaming response
        mock_openai_chunk1 = MagicMock()
        mock_openai_chunk2 = MagicMock()
        mock_stream_completion.return_value = [mock_openai_chunk1, mock_openai_chunk2]

        result = provider._make_api_call("model-id", messages, stream=True)

        # Verify _stream_completion was called
        assert mock_stream_completion.call_count == 1

        # Verify the call arguments (excluding the client object which may be different)
        call_args = mock_stream_completion.call_args
        assert call_args is not None
        args, kwargs = call_args
        assert len(args) >= 3  # client, model, messages
        assert args[1] == "model-id"  # model
        assert args[2] == messages  # messages
        assert kwargs.get("stream") is True

        # Verify the result is a list (since we mocked it to return a list)
        assert isinstance(result, list)
        assert len(result) == 2
