from any_llm import AnyLLM
from any_llm.providers.gmi.gmi import GmiProvider
from any_llm.types.completion import CompletionParams


def test_provider_basics() -> None:
    provider = GmiProvider(api_key="dummy_key")
    assert provider.PROVIDER_NAME == "gmi"
    assert provider.ENV_API_KEY_NAME == "GMI_API_KEY"
    assert provider.ENV_API_BASE_NAME == "GMI_API_BASE"
    assert provider.API_BASE == "https://api.gmi-serving.com/v1"


def test_gmi_supports_flags() -> None:
    provider = GmiProvider(api_key="dummy_key")
    assert provider.SUPPORTS_COMPLETION is True
    assert provider.SUPPORTS_COMPLETION_STREAMING is True
    assert provider.SUPPORTS_COMPLETION_REASONING is True
    assert provider.SUPPORTS_COMPLETION_IMAGE is False
    assert provider.SUPPORTS_COMPLETION_PDF is False
    assert provider.SUPPORTS_EMBEDDING is False
    assert provider.SUPPORTS_MODERATION is False
    assert provider.SUPPORTS_LIST_MODELS is True
    assert provider.SUPPORTS_RESPONSES is False


def test_factory_integration() -> None:
    provider = AnyLLM.create("gmi", api_key="sk-1")
    assert isinstance(provider, GmiProvider)
    assert provider.PROVIDER_NAME == "gmi"
    assert "gmi" in AnyLLM.get_supported_providers()


def test_gmi_remaps_max_tokens_back_to_max_tokens() -> None:
    params = CompletionParams(
        model_id="zai-org/GLM-5-FP8", messages=[{"role": "user", "content": "hi"}], max_tokens=8192
    )
    result = GmiProvider._convert_completion_params(params)
    assert result["max_tokens"] == 8192
    assert "max_completion_tokens" not in result


def test_gmi_remaps_max_completion_tokens_to_max_tokens() -> None:
    params = CompletionParams(
        model_id="zai-org/GLM-5-FP8",
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=4096,
    )
    result = GmiProvider._convert_completion_params(params)
    assert result["max_tokens"] == 4096
    assert "max_completion_tokens" not in result


def test_provider_metadata() -> None:
    metadata = GmiProvider.get_provider_metadata()
    assert metadata.name == "gmi"
    assert metadata.env_key == "GMI_API_KEY"
    assert metadata.env_api_base == "GMI_API_BASE"
    assert metadata.doc_url == "https://docs.gmicloud.ai/inference-engine/api-reference/llm-api-reference"
    assert metadata.completion is True
    assert metadata.reasoning is True
    assert metadata.image is False
    assert metadata.embedding is False
    assert metadata.list_models is True
