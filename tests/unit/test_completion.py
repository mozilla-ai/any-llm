from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm import acompletion
from any_llm.provider import ApiConfig, Provider, ProviderFactory, ProviderName
from any_llm.types.completion import ChatCompletionMessage, CompletionParams, Reasoning


@pytest.mark.asyncio
async def test_completion_invalid_model_format_no_slash() -> None:
    """Test completion raises ValueError for model without separator."""
    with pytest.raises(
        ValueError, match="Invalid model format. Expected 'provider:model' or 'provider/model', got 'gpt-4'"
    ):
        await acompletion("gpt-4", messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_completion_invalid_model_format_empty_provider() -> None:
    """Test completion raises ValueError for model with empty provider."""
    with pytest.raises(ValueError, match="Invalid model format"):
        await acompletion("/model", messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_completion_invalid_model_format_empty_model() -> None:
    """Test completion raises ValueError for model with empty model name."""
    with pytest.raises(ValueError, match="Invalid model format"):
        await acompletion("provider/", messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_completion_invalid_model_format_multiple_slashes() -> None:
    """Test completion handles multiple slashes correctly (should work - takes first split)."""
    mock_provider = Mock()
    mock_provider.acompletion = AsyncMock()

    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["provider"]
        mock_factory.get_provider_enum.return_value = ProviderName.OPENAI  # Using a valid provider
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "model/extra")
        mock_factory.create_provider.return_value = mock_provider

        await acompletion("provider/model/extra", messages=[{"role": "user", "content": "Hello"}])

        mock_provider.acompletion.assert_called_once()
        args, kwargs = mock_provider.acompletion.call_args
        assert isinstance(args[0], CompletionParams)
        assert args[0].model_id == "model/extra"
        assert args[0].messages == [{"role": "user", "content": "Hello"}]
        assert kwargs == {}


@pytest.mark.asyncio
async def test_completion_converts_chat_message_to_dict() -> None:
    mock_provider = Mock()
    mock_provider.acompletion = AsyncMock()

    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["provider"]
        mock_factory.get_provider_enum.return_value = ProviderName.OPENAI
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "gpt-4o")
        mock_factory.create_provider.return_value = mock_provider

        msg = ChatCompletionMessage(role="assistant", content="Hello", reasoning=Reasoning(content="Thinking..."))
        await acompletion("provider/gpt-4o", messages=[msg])

        mock_provider.acompletion.assert_called_once()
        args, _ = mock_provider.acompletion.call_args
        assert isinstance(args[0], CompletionParams)
        # reasoning shouldn't show up because it gets stripped out and only role and content are sent
        assert args[0].messages == [{"role": "assistant", "content": "Hello"}]


@pytest.mark.asyncio
async def test_all_providers_can_be_loaded(provider: str) -> None:
    """Test that all supported providers can be loaded successfully.

    This test uses the provider fixture which iterates over all providers
    returned by ProviderFactory.get_supported_providers(). It verifies that:
    1. Each provider can be imported and instantiated
    2. The created instance is actually a Provider
    3. No ImportError or other exceptions are raised during loading

    This test will automatically include new providers when they're added
    without requiring any code changes.
    """
    provider_instance = ProviderFactory.create_provider(provider, ApiConfig(api_key="test_key"))

    assert isinstance(provider_instance, Provider), f"Provider {provider} did not create a valid Provider instance"

    assert hasattr(provider_instance, "acompletion"), f"Provider {provider} does not have an acompletion method"
    assert callable(provider_instance.acompletion), f"Provider {provider} acompletion is not callable"


@pytest.mark.asyncio
async def test_all_providers_can_be_loaded_with_config(provider: str) -> None:
    """Test that all supported providers can be loaded with sample config parameters.

    This test verifies that providers can handle common configuration parameters
    like api_key and api_base without throwing errors during instantiation.
    """
    sample_config = ApiConfig(api_key="test_key", api_base="https://test.example.com")

    provider_instance = ProviderFactory.create_provider(provider, sample_config)

    assert isinstance(provider_instance, Provider), (
        f"Provider {provider} did not create a valid Provider instance with config"
    )


@pytest.mark.asyncio
async def test_provider_factory_can_create_all_supported_providers() -> None:
    """Test that ProviderFactory can create instances of all providers it claims to support."""
    supported_providers = ProviderFactory.get_supported_providers()

    for provider_name in supported_providers:
        provider_instance = ProviderFactory.create_provider(provider_name, ApiConfig(api_key="test_key"))

        assert isinstance(provider_instance, Provider), f"Failed to create valid Provider instance for {provider_name}"
