from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.providers.azureopenai.azureopenai import AzureopenaiProvider
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_azureopenai_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai_client,
    ):
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        yield mock_client_instance, mock_openai_client


@pytest.mark.asyncio
async def test_azureopenai_default_query_with_existing_kwargs() -> None:
    """Test that AzureopenaiProvider preserves existing kwargs while adding default_query."""
    api_key = "test-api-key"
    api_base = "https://test.openai.azure.com"
    custom_timeout = 30

    messages = [{"role": "user", "content": "Hello"}]

    with mock_azureopenai_provider() as (mock_client, mock_openai_client):
        provider = AzureopenaiProvider(api_key=api_key, api_base=api_base, timeout=custom_timeout)
        await provider._acompletion(CompletionParams(model_id="gpt-4", messages=messages))

        mock_openai_client.assert_called_once()
        call_args = mock_openai_client.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert "default_query" in kwargs
        assert kwargs["default_query"] == {"api-version": "preview"}
        assert kwargs["timeout"] == custom_timeout

        mock_client.chat.completions.create.assert_called_once()
