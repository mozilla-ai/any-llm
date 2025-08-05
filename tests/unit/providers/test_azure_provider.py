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
