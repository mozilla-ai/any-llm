import json
from contextlib import contextmanager
from unittest.mock import patch, MagicMock

from any_llm.provider import ApiConfig
from any_llm.providers.azure.azure import AzureProvider


@contextmanager
def mock_azure_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.azure.azure.urllib.request") as mock_azure,
        patch("any_llm.providers.azure.azure._convert_response"),
    ):
        urlopen = MagicMock()
        urlopen.read.return_value = "{}"
        mock_azure.urlopen.return_value.__enter__.return_value = urlopen
        yield mock_azure


def test_azure_with_api_key_and_api_base() -> None:
    api_key = "test-api-key"
    custom_endpoint = "https://test.eu.models.ai.azure.com"

    messages = [{"role": "user", "content": "Hello"}]
    with mock_azure_provider() as mock_azure:
        provider = AzureProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))
        provider._make_api_call("model-id", messages)

        request_args = (
            f"{custom_endpoint}/chat/completions",
            json.dumps({"messages": messages}).encode("utf-8"),
            {
                "Content-Type": "application/json",
                "Authorization": "test-api-key",
            },
        )
        mock_azure.Request.assert_called_with(*request_args)
        mock_azure.urlopen.assert_called_with(mock_azure.Request(*request_args))


def test_azure_with_tools() -> None:
    api_key = "test-api-key"
    custom_endpoint = "https://aoairesource.openai.azure.com"

    messages = [{"role": "user", "content": "Hello"}]
    tools = {"type": "function", "function": "foo"}
    tool_choice = "auto"
    with mock_azure_provider() as mock_azure:
        provider = AzureProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))
        provider._make_api_call("model-id", messages, tools=tools, tool_choice=tool_choice)

        request_args = (
            f"{custom_endpoint}/chat/completions",
            json.dumps({"messages": messages, "tools": tools, "tool_choice": tool_choice}).encode("utf-8"),
            {
                "Content-Type": "application/json",
                "Authorization": "test-api-key",
            },
        )
        mock_azure.Request.assert_called_with(*request_args)
        mock_azure.urlopen.assert_called_with(mock_azure.Request(*request_args))
