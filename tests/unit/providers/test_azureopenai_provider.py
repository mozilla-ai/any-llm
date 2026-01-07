from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.providers.azureopenai.azureopenai import AzureopenaiProvider
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_azureopenai_provider():  # type: ignore[no-untyped-def]
    with patch("any_llm.providers.azureopenai.azureopenai.AsyncAzureOpenAI") as mock_azure_client:
        mock_client_instance = MagicMock()
        mock_azure_client.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        yield mock_client_instance, mock_azure_client


@pytest.mark.asyncio
async def test_azureopenai_uses_azure_endpoint() -> None:
    """Test that AzureopenaiProvider maps api_base to azure_endpoint."""
    api_key = "test-api-key"
    api_base = "https://test.openai.azure.com"

    messages = [{"role": "user", "content": "Hello"}]

    with mock_azureopenai_provider() as (mock_client, mock_azure_client):
        provider = AzureopenaiProvider(api_key=api_key, api_base=api_base)
        await provider._acompletion(CompletionParams(model_id="gpt-4", messages=messages))

        mock_azure_client.assert_called_once()
        call_args = mock_azure_client.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs["azure_endpoint"] == api_base
        assert kwargs["api_key"] == api_key

        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_azureopenai_api_version_default() -> None:
    """Test that AzureopenaiProvider uses default API version."""
    api_key = "test-api-key"
    api_base = "https://test.openai.azure.com"

    with mock_azureopenai_provider() as (_, mock_azure_client):
        AzureopenaiProvider(api_key=api_key, api_base=api_base)

        mock_azure_client.assert_called_once()
        call_args = mock_azure_client.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs["api_version"] == "2024-02-01"


@pytest.mark.asyncio
async def test_azureopenai_api_version_custom() -> None:
    """Test that AzureopenaiProvider accepts custom API version via kwargs."""
    api_key = "test-api-key"
    api_base = "https://test.openai.azure.com"
    custom_api_version = "2024-06-01"

    with mock_azureopenai_provider() as (_, mock_azure_client):
        AzureopenaiProvider(api_key=api_key, api_base=api_base, api_version=custom_api_version)

        mock_azure_client.assert_called_once()
        call_args = mock_azure_client.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs["api_version"] == custom_api_version


@pytest.mark.asyncio
async def test_azureopenai_api_version_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that AzureopenaiProvider reads API version from environment."""
    api_key = "test-api-key"
    api_base = "https://test.openai.azure.com"
    env_api_version = "2024-08-01-preview"

    monkeypatch.setenv("OPENAI_API_VERSION", env_api_version)

    with mock_azureopenai_provider() as (_, mock_azure_client):
        AzureopenaiProvider(api_key=api_key, api_base=api_base)

        mock_azure_client.assert_called_once()
        call_args = mock_azure_client.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs["api_version"] == env_api_version


@pytest.mark.asyncio
async def test_azureopenai_ad_token_auth() -> None:
    """Test that AzureopenaiProvider supports Azure AD token authentication."""
    api_base = "https://test.openai.azure.com"
    azure_ad_token = "test-ad-token"  # noqa: S105

    with mock_azureopenai_provider() as (_, mock_azure_client):
        AzureopenaiProvider(api_key=None, api_base=api_base, azure_ad_token=azure_ad_token)

        mock_azure_client.assert_called_once()
        call_args = mock_azure_client.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs["azure_ad_token"] == azure_ad_token


@pytest.mark.asyncio
async def test_azureopenai_preserves_extra_kwargs() -> None:
    """Test that AzureopenaiProvider preserves extra kwargs."""
    api_key = "test-api-key"
    api_base = "https://test.openai.azure.com"
    custom_timeout = 30

    with mock_azureopenai_provider() as (_, mock_azure_client):
        AzureopenaiProvider(api_key=api_key, api_base=api_base, timeout=custom_timeout)

        mock_azure_client.assert_called_once()
        call_args = mock_azure_client.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs["timeout"] == custom_timeout
