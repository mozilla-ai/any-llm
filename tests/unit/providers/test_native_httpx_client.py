"""Test http_client support for providers with native SDK support."""

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from any_llm.provider import ApiConfig, ProviderFactory, ProviderName
from any_llm.types.completion import CompletionParams


@pytest.mark.parametrize(
    ("provider", "module_path", "client_param"),
    [
        (ProviderName.ANTHROPIC, "anthropic.Anthropic", "http_client"),
        (ProviderName.GROQ, "groq.Groq", "http_client"),
        (ProviderName.MISTRAL, "mistralai.Mistral", "client"),
        (ProviderName.COHERE, "cohere.ClientV2", "httpx_client"),
    ],
)
def test_native_provider_accepts_http_client(provider: ProviderName, module_path: str, client_param: str) -> None:
    """Test that native SDK providers accept and pass through http_client."""
    mock_http_client = MagicMock(spec=httpx.Client)

    config = ApiConfig(api_key="test-key")
    provider_instance = ProviderFactory.create_provider(provider, config)

    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "test"}],
    )

    # Patch the client constructor to verify http_client is passed
    with patch(module_path) as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Set up mock response based on provider
        if provider == ProviderName.ANTHROPIC:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="test response")]
            mock_response.role = "assistant"
            mock_response.stop_reason = "end_turn"
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response
        elif provider == ProviderName.GROQ:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(
                    message=MagicMock(content="test response", role="assistant", tool_calls=None),
                    finish_reason="stop",
                    index=0,
                )
            ]
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            mock_response.id = "test-id"
            mock_response.model = "test-model"
            mock_response.created = 1234567890
            mock_client.chat.completions.create.return_value = mock_response
        elif provider == ProviderName.MISTRAL:
            mock_response = MagicMock()
            mock_response.id = "test-id"
            mock_response.choices = [
                MagicMock(
                    message=MagicMock(content="test response", role="assistant"),
                    finish_reason="stop",
                )
            ]
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            mock_response.created = 1234567890
            mock_response.model = "test-model"
            mock_client.chat.complete.return_value = mock_response
        elif provider == ProviderName.COHERE:
            mock_response = MagicMock()
            mock_response.message = MagicMock(
                content=[MagicMock(type="text", text="test response")],
                role="assistant",
            )
            mock_response.usage = MagicMock(tokens=MagicMock(input_tokens=10, output_tokens=20))
            mock_response.id = "test-id"
            mock_client.chat.return_value = mock_response

        # Call completion with http_client
        provider_instance.completion(params, http_client=mock_http_client)

        # Verify client was instantiated with http_client
        mock_client_class.assert_called()
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs[client_param] == mock_http_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "module_path", "client_param"),
    [
        (ProviderName.ANTHROPIC, "anthropic.AsyncAnthropic", "http_client"),
        (ProviderName.GROQ, "groq.AsyncGroq", "http_client"),
        (ProviderName.MISTRAL, "mistralai.Mistral", "client"),
        (ProviderName.COHERE, "cohere.AsyncClientV2", "httpx_client"),
    ],
)
async def test_native_provider_accepts_http_client_async(
    provider: ProviderName, module_path: str, client_param: str
) -> None:
    """Test that native SDK providers accept and pass through http_client in async."""
    mock_http_client = MagicMock(spec=httpx.AsyncClient)

    config = ApiConfig(api_key="test-key")
    provider_instance = ProviderFactory.create_provider(provider, config)

    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "test"}],
    )

    # Patch the async client constructor to verify http_client is passed
    with patch(module_path) as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Set up mock response based on provider
        if provider == ProviderName.ANTHROPIC:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="test response")]
            mock_response.role = "assistant"
            mock_response.stop_reason = "end_turn"
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)

            async def async_create(*args: Any, **kwargs: Any) -> Any:
                return mock_response

            mock_client.messages.create = async_create
        elif provider == ProviderName.GROQ:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(
                    message=MagicMock(content="test response", role="assistant", tool_calls=None),
                    finish_reason="stop",
                    index=0,
                )
            ]
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            mock_response.id = "test-id"
            mock_response.model = "test-model"
            mock_response.created = 1234567890

            async def async_create(*args: Any, **kwargs: Any) -> Any:
                return mock_response

            mock_client.chat.completions.create = async_create
        elif provider == ProviderName.MISTRAL:
            mock_response = MagicMock()
            mock_response.id = "test-id"
            mock_response.choices = [
                MagicMock(
                    message=MagicMock(content="test response", role="assistant"),
                    finish_reason="stop",
                )
            ]
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            mock_response.created = 1234567890
            mock_response.model = "test-model"

            async def async_complete(*args: Any, **kwargs: Any) -> Any:
                return mock_response

            mock_client.chat.complete_async = async_complete
        elif provider == ProviderName.COHERE:
            mock_response = MagicMock()
            mock_response.message = MagicMock(
                content=[MagicMock(type="text", text="test response")],
                role="assistant",
            )
            mock_response.usage = MagicMock(tokens=MagicMock(input_tokens=10, output_tokens=20))
            mock_response.id = "test-id"

            async def async_chat(*args: Any, **kwargs: Any) -> Any:
                return mock_response

            mock_client.chat = async_chat

        # Call acompletion with http_client
        await provider_instance.acompletion(params, http_client=mock_http_client)

        # Verify client was instantiated with http_client
        mock_client_class.assert_called()
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs[client_param] == mock_http_client
