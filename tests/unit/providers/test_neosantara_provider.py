from any_llm import AnyLLM
from any_llm.providers.neosantara.neosantara import NeosantaraProvider
from any_llm.types.completion import CompletionParams


def test_provider_basics() -> None:
    provider = NeosantaraProvider(api_key="dummy_key")
    assert provider.PROVIDER_NAME == "neosantara"
    assert provider.ENV_API_KEY_NAME == "NEOSANTARA_API_KEY"
    assert provider.ENV_API_BASE_NAME == "NEOSANTARA_API_BASE"
    assert provider.API_BASE == "https://api.neosantara.xyz/v1"


def test_neosantara_supports_flags() -> None:
    provider = NeosantaraProvider(api_key="dummy_key")
    assert provider.SUPPORTS_COMPLETION is True
    assert provider.SUPPORTS_COMPLETION_STREAMING is True
    assert provider.SUPPORTS_RESPONSES is True
    assert provider.SUPPORTS_COMPLETION_REASONING is True
    assert provider.SUPPORTS_COMPLETION_IMAGE is True
    assert provider.SUPPORTS_COMPLETION_PDF is True
    assert provider.SUPPORTS_EMBEDDING is True
    assert provider.SUPPORTS_IMAGE_GENERATION is True
    assert provider.SUPPORTS_LIST_MODELS is True


def test_factory_integration() -> None:
    provider = AnyLLM.create("neosantara", api_key="nsk_test")
    assert isinstance(provider, NeosantaraProvider)
    assert provider.PROVIDER_NAME == "neosantara"
    assert "neosantara" in AnyLLM.get_supported_providers()


def test_neosantara_uses_openai_compatible_token_conversion() -> None:
    params = CompletionParams(
        model_id="garda-core",
        messages=[{"role": "user", "content": "Halo"}],
        max_tokens=100,
    )
    result = NeosantaraProvider._convert_completion_params(params)
    assert result["max_completion_tokens"] == 100
    assert "max_tokens" not in result


def test_provider_metadata() -> None:
    metadata = NeosantaraProvider.get_provider_metadata()
    assert metadata.name == "neosantara"
    assert metadata.env_key == "NEOSANTARA_API_KEY"
    assert metadata.env_api_base == "NEOSANTARA_API_BASE"
    assert metadata.doc_url == "https://docs.neosantara.xyz"
    assert metadata.completion is True
