from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm import AnyLLM
from any_llm.api import acompletion
from any_llm.constants import LLMProvider


@pytest.mark.asyncio
async def test_completion_invalid_model_format_no_slash() -> None:
    """Test completion raises ValueError for model without separator."""
    with pytest.raises(
        ValueError, match=r"Invalid model format. Expected 'provider:model' or 'provider/model', got 'gpt-4'"
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

    with (
        patch("any_llm.any_llm.AnyLLM.split_model_provider") as mock_split,
        patch("any_llm.any_llm.AnyLLM.create") as mock_create,
    ):
        mock_split.return_value = (LLMProvider.OPENAI, "model/extra")
        mock_create.return_value = mock_provider

        await acompletion("openai/model/extra", messages=[{"role": "user", "content": "Hello"}])

        mock_provider.acompletion.assert_called_once()
        _, kwargs = mock_provider.acompletion.call_args
        assert kwargs["model"] == "model/extra"


@pytest.mark.asyncio
async def test_all_providers_can_be_loaded(provider: str) -> None:
    """Test that all supported providers can be loaded successfully.

    This test uses the provider fixture which iterates over all providers
    returned by AnyLLM.get_supported_providers(). It verifies that:
    1. Each provider can be imported and instantiated
    2. The created instance is actually a AnyLLM
    3. No ImportError or other exceptions are raised during loading

    This test will automatically include new providers when they're added
    without requiring any code changes.
    """
    kwargs: dict[str, Any] = {"api_key": "test_key"}
    if provider == "azure":
        kwargs["api_base"] = "test_api_base"
    if provider == "bedrock":
        kwargs["region_name"] = "us-east-1"
    if provider == "sagemaker":
        pytest.skip("sagemaker requires AWS credentials on instantiation")
    if provider == "vertexai":
        kwargs["project"] = "test-project"
        kwargs["location"] = "test-location"

    provider_instance = AnyLLM.create(provider, **kwargs)

    assert isinstance(provider_instance, AnyLLM), f"Provider {provider} did not create a valid AnyLLM instance"

    assert hasattr(provider_instance, "acompletion"), f"Provider {provider} does not have an acompletion method"
    assert callable(provider_instance._acompletion), f"Provider {provider} acompletion is not callable"


@pytest.mark.asyncio
async def test_all_providers_can_be_loaded_with_config(provider: str) -> None:
    """Test that all supported providers can be loaded with sample config parameters.

    This test verifies that providers can handle common configuration parameters
    like api_key and api_base without throwing errors during instantiation.
    """
    kwargs: dict[str, Any] = {"api_key": "test_key", "api_base": "https://test.example.com"}
    if provider == "bedrock":
        kwargs["region_name"] = "us-east-1"
    if provider == "vertexai":
        kwargs["project"] = "test-project"
        kwargs["location"] = "test-location"
    if provider == "sagemaker":
        pytest.skip("sagemaker requires AWS credentials on instantiation")

    provider_instance = AnyLLM.create(provider, **kwargs)

    assert isinstance(provider_instance, AnyLLM), (
        f"Provider {provider} did not create a valid AnyLLM instance with config"
    )


@pytest.mark.asyncio
async def test_provider_factory_can_create_all_supported_providers() -> None:
    """Test that AnyLLM can create instances of all providers it claims to support."""
    supported_providers = AnyLLM.get_supported_providers()

    for provider_name in supported_providers:
        kwargs: dict[str, Any] = {"api_key": "test_key"}
        if provider_name == "azure":
            kwargs["api_base"] = "test_api_base"
        if provider_name == "bedrock":
            kwargs["region_name"] = "us-east-1"
        if provider_name == "vertexai":
            kwargs["project"] = "test-project"
            kwargs["location"] = "test-location"
        if provider_name == "sagemaker":
            continue
        provider_instance = AnyLLM.create(provider_name, **kwargs)

        assert isinstance(provider_instance, AnyLLM), f"Failed to create valid AnyLLM instance for {provider_name}"
