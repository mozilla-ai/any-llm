from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.providers.azureanthropic.azureanthropic import AzureanthropicProvider
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_azureanthropic_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.azureanthropic.azureanthropic.AsyncAnthropicFoundry") as mock_foundry,
        patch("any_llm.providers.anthropic.base._convert_response"),
    ):
        mock_client = Mock()
        mock_foundry.return_value = mock_client
        mock_client.messages.create = AsyncMock()
        yield mock_foundry


@pytest.mark.asyncio
async def test_azureanthropic_client_created_with_api_key_and_resource() -> None:
    api_key = "test-key"
    resource = "my-foundry-resource"

    with mock_azureanthropic_provider() as mock_foundry:
        provider = AzureanthropicProvider(api_key=api_key, resource=resource)
        await provider._acompletion(
            CompletionParams(model_id="claude-sonnet-4-5-20250514", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_foundry.assert_called_once_with(resource=resource, api_key=api_key)


@pytest.mark.asyncio
async def test_azureanthropic_client_uses_env_vars() -> None:
    with (
        mock_azureanthropic_provider() as mock_foundry,
        patch.dict(
            "os.environ",
            {
                "AZURE_ANTHROPIC_API_KEY": "env-key",
                "AZURE_ANTHROPIC_RESOURCE": "env-resource",
            },
        ),
    ):
        provider = AzureanthropicProvider()
        await provider._acompletion(
            CompletionParams(model_id="claude-sonnet-4-5-20250514", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_foundry.assert_called_once_with(resource="env-resource", api_key="env-key")


@pytest.mark.asyncio
async def test_azureanthropic_client_without_resource() -> None:
    api_key = "test-key"

    with mock_azureanthropic_provider() as mock_foundry:
        provider = AzureanthropicProvider(api_key=api_key)
        await provider._acompletion(
            CompletionParams(model_id="claude-sonnet-4-5-20250514", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_foundry.assert_called_once_with(resource=None, api_key=api_key)


@pytest.mark.asyncio
async def test_azureanthropic_client_with_api_base() -> None:
    api_key = "test-key"

    with (
        mock_azureanthropic_provider() as mock_foundry,
        patch.dict("os.environ", {"AZURE_ANTHROPIC_API_BASE": "https://custom.azure.endpoint"}),
    ):
        provider = AzureanthropicProvider(api_key=api_key)
        await provider._acompletion(
            CompletionParams(model_id="claude-sonnet-4-5-20250514", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_foundry.assert_called_once_with(base_url="https://custom.azure.endpoint", api_key=api_key)


@pytest.mark.asyncio
async def test_azureanthropic_constructor_arg_overrides_env_var() -> None:
    with (
        mock_azureanthropic_provider() as mock_foundry,
        patch.dict(
            "os.environ",
            {
                "AZURE_ANTHROPIC_API_KEY": "env-key",
                "AZURE_ANTHROPIC_RESOURCE": "env-resource",
            },
        ),
    ):
        provider = AzureanthropicProvider(api_key="arg-key", resource="arg-resource")
        await provider._acompletion(
            CompletionParams(model_id="claude-sonnet-4-5-20250514", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_foundry.assert_called_once_with(resource="arg-resource", api_key="arg-key")


@pytest.mark.asyncio
async def test_azureanthropic_completion_calls_messages_create() -> None:
    api_key = "test-key"
    model = "claude-sonnet-4-5-20250514"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_azureanthropic_provider() as mock_foundry:
        provider = AzureanthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        mock_foundry.return_value.messages.create.assert_called_once()
        call_kwargs = mock_foundry.return_value.messages.create.call_args[1]
        assert call_kwargs["model"] == model
        assert call_kwargs["messages"] == messages


@pytest.mark.asyncio
async def test_azureanthropic_completion_with_system_message() -> None:
    api_key = "test-key"
    model = "claude-sonnet-4-5-20250514"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    with mock_azureanthropic_provider() as mock_foundry:
        provider = AzureanthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        call_kwargs = mock_foundry.return_value.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant."
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]


def test_azureanthropic_provider_name() -> None:
    with mock_azureanthropic_provider():
        provider = AzureanthropicProvider(api_key="test-key")
        assert provider.PROVIDER_NAME == "azureanthropic"


def test_azureanthropic_env_api_key_name() -> None:
    with mock_azureanthropic_provider():
        provider = AzureanthropicProvider(api_key="test-key")
        assert provider.ENV_API_KEY_NAME == "AZURE_ANTHROPIC_API_KEY"


def test_azureanthropic_does_not_support_list_models() -> None:
    with mock_azureanthropic_provider():
        provider = AzureanthropicProvider(api_key="test-key")
        assert provider.SUPPORTS_LIST_MODELS is False


def test_azureanthropic_supports_completion() -> None:
    with mock_azureanthropic_provider():
        provider = AzureanthropicProvider(api_key="test-key")
        assert provider.SUPPORTS_COMPLETION is True
        assert provider.SUPPORTS_COMPLETION_STREAMING is True
        assert provider.SUPPORTS_COMPLETION_REASONING is True
        assert provider.SUPPORTS_COMPLETION_IMAGE is True
