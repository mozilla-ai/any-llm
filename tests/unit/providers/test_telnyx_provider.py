from any_llm import AnyLLM
from any_llm.providers.telnyx.telnyx import TelnyxProvider
from any_llm.types.completion import CompletionParams


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    provider = TelnyxProvider(api_key="dummy_key")
    assert provider.PROVIDER_NAME == "telnyx"
    assert provider.ENV_API_KEY_NAME == "TELNYX_API_KEY"
    assert provider.ENV_API_BASE_NAME == "TELNYX_API_BASE"
    assert provider.API_BASE == "https://api.telnyx.com/v2/ai"


def test_telnyx_supports_flags() -> None:
    """Test that TelnyxProvider has correct feature support flags."""
    provider = TelnyxProvider(api_key="dummy_key")
    assert provider.SUPPORTS_COMPLETION is True
    assert provider.SUPPORTS_COMPLETION_STREAMING is True
    assert provider.SUPPORTS_COMPLETION_REASONING is True
    assert provider.SUPPORTS_COMPLETION_PDF is False
    assert provider.SUPPORTS_LIST_MODELS is True


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    provider = AnyLLM.create("telnyx", api_key="sk-1")
    assert isinstance(provider, TelnyxProvider)
    assert provider.PROVIDER_NAME == "telnyx"
    assert "telnyx" in AnyLLM.get_supported_providers()


def test_telnyx_convert_completion_params_renames_max_tokens() -> None:
    """max_tokens should be remapped to max_completion_tokens by the base provider."""
    params = CompletionParams(
        model_id="moonshotai/Kimi-K2.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )
    result = TelnyxProvider._convert_completion_params(params)
    assert result["max_completion_tokens"] == 100
    assert "max_tokens" not in result


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = TelnyxProvider.get_provider_metadata()
    assert metadata.name == "telnyx"
    assert metadata.env_key == "TELNYX_API_KEY"
    assert metadata.env_api_base == "TELNYX_API_BASE"
    assert metadata.doc_url == "https://developers.telnyx.com/docs/inference/getting-started"
    assert metadata.completion is True
