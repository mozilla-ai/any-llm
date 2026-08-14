import dataclasses
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.ai.inference.models import JsonSchemaFormat, ModelInfo
from azure.core.exceptions import HttpResponseError

from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.azure.azure import AzureProvider
from any_llm.providers.azure.utils import _convert_model_info_to_list, _convert_response_format
from any_llm.types.completion import CompletionParams
from any_llm.types.model import Model


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
        provider = AzureProvider(api_key=api_key, api_base=custom_endpoint)
        await provider._acompletion(CompletionParams(model_id="model-id", messages=messages))

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
            provider = AzureProvider(api_key=api_key, api_base=custom_endpoint, api_version="2025-04-01-preview")
            await provider._acompletion(
                CompletionParams(model_id="model-id", messages=messages),
            )

            mock_chat_client.assert_called_once_with(
                endpoint=custom_endpoint,
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
    with mock_azure_provider() as (mock_client, mock_convert_response, _):
        provider = AzureProvider(api_key=api_key, api_base=custom_endpoint)
        await provider._acompletion(
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

    provider = AzureProvider(api_key=api_key, api_base=custom_endpoint)

    with patch.object(provider, "_stream_completion_async") as mock_stream_completion:
        mock_openai_chunk1 = MagicMock()
        mock_openai_chunk2 = MagicMock()
        mock_stream_completion.return_value = [mock_openai_chunk1, mock_openai_chunk2]

        result = await provider._acompletion(CompletionParams(model_id="model-id", messages=messages, stream=True))

        assert mock_stream_completion.call_count == 1

        call_args = mock_stream_completion.call_args
        assert call_args is not None
        args, kwargs = call_args
        assert len(args) == 2  # model, messages
        assert args[0] == "model-id"  # model
        assert args[1] == messages  # messages
        assert kwargs.get("stream") is True

        assert isinstance(result, list)
        assert len(result) == 2


def test_convert_response_format_from_dict() -> None:
    response_format_dict = {
        "json_schema": {
            "name": "TestSchema",
            "schema": {"type": "object", "properties": {"field": {"type": "string"}}},
            "description": "A test schema",
            "strict": True,
        }
    }

    result = _convert_response_format(response_format_dict)

    assert isinstance(result, JsonSchemaFormat)
    assert result.name == "TestSchema"
    assert result.schema == {"type": "object", "properties": {"field": {"type": "string"}}}
    assert result.description == "A test schema"
    assert result.strict is True


def test_convert_response_format_from_dataclass() -> None:
    @dataclasses.dataclass
    class CityResponse:
        city_name: str

    result = _convert_response_format(CityResponse)

    assert isinstance(result, JsonSchemaFormat)
    assert result.name == "CityResponse"
    assert "city_name" in result.schema["properties"]
    assert result.schema["additionalProperties"] is False
    assert result.strict is True


def test_convert_response_format_from_dict_invalid() -> None:
    invalid_dict = {"type": "json_object"}

    with pytest.raises(ValueError, match="Response format must be a structured type or a dict with a json_schema key"):
        _convert_response_format(invalid_dict)


@pytest.mark.asyncio
async def test_azure_with_token_credential() -> None:
    """Test that AzureProvider accepts a TokenCredential for Entra ID auth."""
    custom_endpoint = "https://test.eu.models.ai.azure.com"
    mock_credential = MagicMock()

    messages = [{"role": "user", "content": "Hello"}]
    with mock_azure_provider() as (mock_client, mock_convert_response, mock_chat_client):
        provider = AzureProvider(api_base=custom_endpoint, credential=mock_credential)
        await provider._acompletion(CompletionParams(model_id="model-id", messages=messages))

        mock_chat_client.assert_called_once_with(
            endpoint=custom_endpoint,
            credential=mock_credential,
        )

        mock_client.complete.assert_called_once_with(
            model="model-id",
            messages=messages,
        )

        mock_convert_response.assert_called_once_with(mock_client.complete.return_value)


@pytest.mark.asyncio
async def test_azure_token_credential_ignores_api_key() -> None:
    """Test that when a credential is provided, api_key is ignored."""
    custom_endpoint = "https://test.eu.models.ai.azure.com"
    mock_credential = MagicMock()

    with mock_azure_provider() as (_, _, mock_chat_client):
        AzureProvider(api_key="should-be-ignored", api_base=custom_endpoint, credential=mock_credential)

        mock_chat_client.assert_called_once_with(
            endpoint=custom_endpoint,
            credential=mock_credential,
        )


def test_azure_no_api_key_no_credential_raises() -> None:
    """Test that omitting both api_key and credential raises MissingApiKeyError."""
    custom_endpoint = "https://test.eu.models.ai.azure.com"

    with mock_azure_provider():
        with pytest.raises(MissingApiKeyError):
            AzureProvider(api_base=custom_endpoint)


def test_convert_completion_params_drops_stream_options() -> None:
    """stream_options is an OpenAI-only knob (set by the Messages bridge for
    streaming usage); the Azure AI Inference SDK does not model it and forwards
    unknown kwargs to the transport, which rejects them, so it must be dropped."""
    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    result = AzureProvider._convert_completion_params(params)
    assert "stream_options" not in result


def test_supports_list_models_flag() -> None:
    """Azure exposes list_models via get_model_info(), so the capability flag is True."""
    assert AzureProvider.SUPPORTS_LIST_MODELS is True


def test_convert_model_info_to_list() -> None:
    """Azure's /info route describes the single model behind the endpoint, so the OpenAI-shaped
    list_models() result is always exactly one item, built from that model's name and provider."""
    model_info = ModelInfo(
        model_name="Phi-4",
        model_type="chat_completion",
        model_provider_name="Microsoft Research",
    )

    result = _convert_model_info_to_list(model_info)

    assert result == [
        Model(id="Phi-4", object="model", created=0, owned_by="Microsoft Research"),
    ]


@pytest.mark.asyncio
async def test_alist_models_calls_get_model_info_on_the_chat_client() -> None:
    """_alist_models must go through get_model_info(), the only model-info route the Azure AI
    Inference SDK exposes (there is no catalog-listing endpoint)."""
    custom_endpoint = "https://test.eu.models.ai.azure.com"

    with mock_azure_provider() as (mock_client_instance, _, _):
        model_info = ModelInfo(
            model_name="Phi-4",
            model_type="chat_completion",
            model_provider_name="Microsoft Research",
        )
        mock_client_instance.get_model_info = AsyncMock(return_value=model_info)

        provider = AzureProvider(api_key="test-key", api_base=custom_endpoint)
        result = await provider._alist_models()

        mock_client_instance.get_model_info.assert_called_once()
        assert result == [
            Model(id="Phi-4", object="model", created=0, owned_by="Microsoft Research"),
        ]


@pytest.mark.asyncio
async def test_alist_models_propagates_http_response_error_for_unsupported_endpoints() -> None:
    """get_model_info() raises HttpResponseError for GitHub Models and Azure OpenAI endpoints
    (it only works for Serverless API / Managed Compute). _alist_models must let that error
    propagate unchanged rather than swallowing it into an empty list."""
    custom_endpoint = "https://test.eu.models.ai.azure.com"

    with mock_azure_provider() as (mock_client_instance, _, _):
        mock_client_instance.get_model_info = AsyncMock(
            side_effect=HttpResponseError(message="This method is not supported for the provided endpoint.")
        )

        provider = AzureProvider(api_key="test-key", api_base=custom_endpoint)

        with pytest.raises(HttpResponseError):
            await provider._alist_models()
