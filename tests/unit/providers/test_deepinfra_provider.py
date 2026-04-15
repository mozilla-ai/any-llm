from any_llm import AnyLLM
from any_llm.providers.deepinfra.deepinfra import DeepinfraProvider
from any_llm.types.completion import CompletionParams


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    provider = DeepinfraProvider(api_key="dummy_key")
    assert provider.PROVIDER_NAME == "deepinfra"
    assert provider.ENV_API_KEY_NAME == "DEEPINFRA_API_KEY"
    assert provider.ENV_API_BASE_NAME == "DEEPINFRA_API_BASE"
    assert provider.API_BASE == "https://api.deepinfra.com/v1/openai"


def test_deepinfra_supports_flags() -> None:
    """Test that DeepinfraProvider has correct feature support flags."""
    provider = DeepinfraProvider(api_key="dummy_key")
    assert provider.SUPPORTS_COMPLETION is True
    assert provider.SUPPORTS_COMPLETION_STREAMING is True
    assert provider.SUPPORTS_COMPLETION_REASONING is True
    assert provider.SUPPORTS_COMPLETION_IMAGE is True
    assert provider.SUPPORTS_COMPLETION_PDF is True
    assert provider.SUPPORTS_EMBEDDING is True
    assert provider.SUPPORTS_LIST_MODELS is True


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    provider = AnyLLM.create("deepinfra", api_key="sk-1")
    assert isinstance(provider, DeepinfraProvider)
    assert provider.PROVIDER_NAME == "deepinfra"
    assert "deepinfra" in AnyLLM.get_supported_providers()


def test_deepinfra_convert_completion_params_renames_max_tokens() -> None:
    """max_tokens should be remapped to max_completion_tokens by the base provider."""
    params = CompletionParams(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )
    result = DeepinfraProvider._convert_completion_params(params)
    assert result["max_completion_tokens"] == 100
    assert "max_tokens" not in result


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = DeepinfraProvider.get_provider_metadata()
    assert metadata.name == "deepinfra"
    assert metadata.env_key == "DEEPINFRA_API_KEY"
    assert metadata.env_api_base == "DEEPINFRA_API_BASE"
    assert metadata.doc_url == "https://deepinfra.com/docs/openai_api"
    assert metadata.completion is True