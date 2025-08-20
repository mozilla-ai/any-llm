"""Test http_client support for OpenAI providers."""

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from any_llm.provider import ApiConfig, ProviderFactory, ProviderName
from any_llm.types.completion import CompletionParams


@pytest.mark.parametrize(
    "provider",
    [
        ProviderName.OPENAI,
        ProviderName.DATABRICKS,
        ProviderName.DEEPSEEK,
        ProviderName.INCEPTION,
        ProviderName.LLAMA,
        ProviderName.LLAMACPP,
        ProviderName.LLAMAFILE,
        ProviderName.LMSTUDIO,
        ProviderName.MOONSHOT,
        ProviderName.NEBIUS,
        ProviderName.OPENROUTER,
        ProviderName.PORTKEY,
        # ProviderName.SAMBANOVA,  # SambaNova overrides completion and needs special handling
    ],
)
def test_openai_provider_accepts_http_client(provider: ProviderName) -> None:
    """Test that OpenAI-based providers accept and pass through http_client."""
    mock_http_client = MagicMock(spec=httpx.Client)

    config = ApiConfig(api_key="test-key")
    provider_instance = ProviderFactory.create_provider(provider, config)

    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "test"}],
    )

    # Patch the OpenAI client constructor to verify http_client is passed
    with patch("any_llm.providers.openai.base.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Set up mock response
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "test-id",
            "created": 1234567890,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_client.chat.completions.create.return_value = mock_response

        # Call completion with http_client
        provider_instance.completion(params, http_client=mock_http_client)

        # Verify OpenAI client was instantiated with http_client
        mock_openai.assert_called()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["http_client"] == mock_http_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    [
        ProviderName.OPENAI,
        ProviderName.DATABRICKS,
        ProviderName.DEEPSEEK,
        ProviderName.INCEPTION,
        ProviderName.LLAMA,
        ProviderName.LLAMACPP,
        ProviderName.LLAMAFILE,
        ProviderName.LMSTUDIO,
        ProviderName.MOONSHOT,
        ProviderName.NEBIUS,
        ProviderName.OPENROUTER,
        ProviderName.PORTKEY,
        # ProviderName.SAMBANOVA,  # SambaNova overrides completion and needs special handling
    ],
)
async def test_openai_provider_accepts_http_client_async(provider: ProviderName) -> None:
    """Test that OpenAI-based providers accept and pass through http_client in async."""
    mock_http_client = MagicMock(spec=httpx.AsyncClient)

    config = ApiConfig(api_key="test-key")
    provider_instance = ProviderFactory.create_provider(provider, config)

    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "test"}],
    )

    # Patch the AsyncOpenAI client constructor to verify http_client is passed
    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_async_openai:
        mock_client = MagicMock()
        mock_async_openai.return_value = mock_client

        # Set up mock response
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "test-id",
            "created": 1234567890,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        async def async_create(*args: Any, **kwargs: Any) -> Any:
            return mock_response

        mock_client.chat.completions.create = async_create

        # Call acompletion with http_client
        await provider_instance.acompletion(params, http_client=mock_http_client)

        # Verify AsyncOpenAI client was instantiated with http_client
        mock_async_openai.assert_called()
        call_kwargs = mock_async_openai.call_args.kwargs
        assert call_kwargs["http_client"] == mock_http_client
