from any_llm import AnyLLM
from any_llm.providers.zyphra.zyphra import ZyphraProvider
from any_llm.types.completion import CompletionParams


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    provider = ZyphraProvider(api_key="dummy_key")
    assert provider.PROVIDER_NAME == "zyphra"
    assert provider.ENV_API_KEY_NAME == "ZYPHRA_API_KEY"
    assert provider.ENV_API_BASE_NAME == "ZYPHRA_API_BASE"
    assert provider.API_BASE == "https://uyppidoc.zyphracloud.com/v1"


def test_zyphra_supports_flags() -> None:
    """Test that ZyphraProvider has correct feature support flags."""
    provider = ZyphraProvider(api_key="dummy_key")
    assert provider.SUPPORTS_COMPLETION is True
    assert provider.SUPPORTS_COMPLETION_STREAMING is True
    assert provider.SUPPORTS_COMPLETION_REASONING is True
    assert provider.SUPPORTS_COMPLETION_IMAGE is False
    assert provider.SUPPORTS_COMPLETION_PDF is False
    assert provider.SUPPORTS_LIST_MODELS is False


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    provider = AnyLLM.create("zyphra", api_key="sk-1")
    assert isinstance(provider, ZyphraProvider)
    assert provider.PROVIDER_NAME == "zyphra"
    assert "zyphra" in AnyLLM.get_supported_providers()


def test_zyphra_convert_completion_params_renames_max_tokens() -> None:
    """max_tokens should be remapped to max_completion_tokens by the base provider."""
    params = CompletionParams(
        model_id="deepseek-ai/DeepSeek-R1-0528",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )
    result = ZyphraProvider._convert_completion_params(params)
    assert result["max_completion_tokens"] == 100
    assert "max_tokens" not in result


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = ZyphraProvider.get_provider_metadata()
    assert metadata.name == "zyphra"
    assert metadata.env_key == "ZYPHRA_API_KEY"
    assert metadata.env_api_base == "ZYPHRA_API_BASE"
    assert metadata.completion is True
