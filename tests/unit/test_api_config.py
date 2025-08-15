from typing import Any
from unittest.mock import Mock, patch

from any_llm import completion
from any_llm.provider import ApiConfig, ProviderName
from any_llm.types.completion import CompletionParams


def test_completion_extracts_all_config_from_kwargs() -> None:
    """Test that api_key and api_base are properly extracted from kwargs to create config."""
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["mistral"]
        mock_factory.get_provider_enum.return_value = ProviderName.MISTRAL
        mock_factory.split_model_provider.return_value = (ProviderName.MISTRAL, "mistral-small")
        mock_factory.create_provider.return_value = mock_provider
        kwargs: dict[str, Any] = {
            "other_param": "value",
        }
        completion(
            model="mistral/mistral-small",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            api_base="https://test.com",
            **kwargs,
        )

        mock_factory.create_provider.assert_called_once_with(
            ProviderName.MISTRAL, ApiConfig(api_key="test_key", api_base="https://test.com")
        )

        mock_provider.completion.assert_called_once()
        args, kwargs = mock_provider.completion.call_args
        assert isinstance(args[0], CompletionParams)
        assert args[0].model_id == "mistral-small"
        assert args[0].messages == [{"role": "user", "content": "Hello"}]
        assert kwargs == {"other_param": "value"}


def test_completion_extracts_partial_config_from_kwargs() -> None:
    """Test that only present config parameters are extracted."""
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["mistral"]
        mock_factory.get_provider_enum.return_value = ProviderName.MISTRAL
        mock_factory.split_model_provider.return_value = (ProviderName.MISTRAL, "mistral-small")
        mock_factory.create_provider.return_value = mock_provider

        completion(
            model="mistral/mistral-small",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            other_param="value",
        )

        mock_factory.create_provider.assert_called_once_with(ProviderName.MISTRAL, ApiConfig(api_key="test_key"))

        mock_provider.completion.assert_called_once()
        args, kwargs = mock_provider.completion.call_args
        assert isinstance(args[0], CompletionParams)
        assert args[0].model_id == "mistral-small"
        assert args[0].messages == [{"role": "user", "content": "Hello"}]
        assert kwargs == {"other_param": "value"}


def test_completion_no_config_extraction() -> None:
    """Test that empty config is created when no config parameters are provided."""
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["mistral"]
        mock_factory.get_provider_enum.return_value = ProviderName.MISTRAL
        mock_factory.split_model_provider.return_value = (ProviderName.MISTRAL, "mistral-small")
        mock_factory.create_provider.return_value = mock_provider

        kwargs: dict[str, Any] = {
            "other_param": "value",
        }
        completion(model="mistral/mistral-small", messages=[{"role": "user", "content": "Hello"}], **kwargs)

        mock_factory.create_provider.assert_called_once_with(ProviderName.MISTRAL, ApiConfig())

        mock_provider.completion.assert_called_once()
        args, kwargs = mock_provider.completion.call_args
        assert isinstance(args[0], CompletionParams)
        assert args[0].model_id == "mistral-small"
        assert args[0].messages == [{"role": "user", "content": "Hello"}]
        assert kwargs == {"other_param": "value"}


def test_completion_extracts_api_base_only() -> None:
    """Test that only api_base is extracted when only it's provided."""
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["ollama"]
        mock_factory.get_provider_enum.return_value = ProviderName.OLLAMA
        mock_factory.split_model_provider.return_value = (ProviderName.OLLAMA, "llama2")
        mock_factory.create_provider.return_value = mock_provider

        completion(
            model="ollama/llama2",
            messages=[{"role": "user", "content": "Test"}],
            api_base="https://custom-endpoint.com",
        )

        mock_factory.create_provider.assert_called_once_with(
            ProviderName.OLLAMA, ApiConfig(api_base="https://custom-endpoint.com")
        )

        mock_provider.completion.assert_called_once()
        args, kwargs = mock_provider.completion.call_args
        assert isinstance(args[0], CompletionParams)
        assert args[0].model_id == "llama2"
        assert args[0].messages == [{"role": "user", "content": "Test"}]
        assert kwargs == {}
