from unittest.mock import Mock, patch
from any_llm import completion
from any_llm.provider import ApiConfig, ProviderName


def test_completion_extracts_all_config_from_kwargs() -> None:
    """Test that api_key and api_base are properly extracted from kwargs to create config."""
    # Mock the provider factory and provider
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with patch("any_llm.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["mistral"]
        mock_factory.get_provider_enum.return_value = ProviderName.MISTRAL
        mock_factory.split_model_provider.return_value = (ProviderName.MISTRAL, "mistral-small")
        mock_factory.create_provider.return_value = mock_provider

        # Test with all config parameters
        completion(
            model="mistral/mistral-small",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            api_base="https://test.com",
            other_param="value",
        )

        # Verify that create_provider was called with extracted config
        mock_factory.create_provider.assert_called_once_with(
            ProviderName.MISTRAL, ApiConfig(api_key="test_key", api_base="https://test.com")
        )

        # Verify that provider.completion was called with remaining kwargs (config keys removed)
        mock_provider.completion.assert_called_once_with(
            "mistral-small", [{"role": "user", "content": "Hello"}], other_param="value"
        )


def test_completion_extracts_partial_config_from_kwargs() -> None:
    """Test that only present config parameters are extracted."""
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with patch("any_llm.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["mistral"]
        mock_factory.get_provider_enum.return_value = ProviderName.MISTRAL
        mock_factory.split_model_provider.return_value = (ProviderName.MISTRAL, "mistral-small")
        mock_factory.create_provider.return_value = mock_provider

        # Test with only api_key
        completion(
            model="mistral/mistral-small",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            other_param="value",
        )

        # Verify that create_provider was called with only api_key in config
        mock_factory.create_provider.assert_called_once_with(ProviderName.MISTRAL, ApiConfig(api_key="test_key"))

        # Verify that provider.completion was called with remaining kwargs
        mock_provider.completion.assert_called_once_with(
            "mistral-small", [{"role": "user", "content": "Hello"}], other_param="value"
        )


def test_completion_no_config_extraction() -> None:
    """Test that empty config is created when no config parameters are provided."""
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with patch("any_llm.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["mistral"]
        mock_factory.get_provider_enum.return_value = ProviderName.MISTRAL
        mock_factory.split_model_provider.return_value = (ProviderName.MISTRAL, "mistral-small")
        mock_factory.create_provider.return_value = mock_provider

        # Test with no config parameters
        completion(model="mistral/mistral-small", messages=[{"role": "user", "content": "Hello"}], other_param="value")

        # Verify that create_provider was called with empty config
        mock_factory.create_provider.assert_called_once_with(ProviderName.MISTRAL, ApiConfig())

        # Verify that provider.completion was called with all kwargs
        mock_provider.completion.assert_called_once_with(
            "mistral-small", [{"role": "user", "content": "Hello"}], other_param="value"
        )


def test_completion_extracts_api_base_only() -> None:
    """Test that only api_base is extracted when only it's provided."""
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with patch("any_llm.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["ollama"]
        mock_factory.get_provider_enum.return_value = ProviderName.OLLAMA
        mock_factory.split_model_provider.return_value = (ProviderName.OLLAMA, "llama2")
        mock_factory.create_provider.return_value = mock_provider

        # Test with only api_base
        completion(
            model="ollama/llama2",
            messages=[{"role": "user", "content": "Test"}],
            api_base="https://custom-endpoint.com",
            temperature=0.7,
        )

        # Verify that create_provider was called with only api_base in config
        mock_factory.create_provider.assert_called_once_with(
            ProviderName.OLLAMA, ApiConfig(api_base="https://custom-endpoint.com")
        )

        # Verify that provider.completion was called with remaining kwargs
        mock_provider.completion.assert_called_once_with(
            "llama2", [{"role": "user", "content": "Test"}], temperature=0.7
        )
