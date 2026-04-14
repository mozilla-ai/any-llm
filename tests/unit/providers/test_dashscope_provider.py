from any_llm import AnyLLM
from any_llm.providers.dashscope.dashscope import DashscopeProvider
from any_llm.types.completion import CompletionParams


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    provider = DashscopeProvider(api_key="dummy_key")
    assert provider.PROVIDER_NAME == "dashscope"
    assert provider.ENV_API_KEY_NAME == "DASHSCOPE_API_KEY"
    assert provider.ENV_API_BASE_NAME == "DASHSCOPE_API_BASE"
    assert provider.API_BASE == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_dashscope_supports_flags() -> None:
    """Test that DashscopeProvider has correct feature support flags."""
    provider = DashscopeProvider(api_key="dummy_key")
    assert provider.SUPPORTS_COMPLETION is True
    assert provider.SUPPORTS_COMPLETION_STREAMING is True
    assert provider.SUPPORTS_COMPLETION_REASONING is False
    assert provider.SUPPORTS_COMPLETION_IMAGE is True
    assert provider.SUPPORTS_COMPLETION_PDF is True
    assert provider.SUPPORTS_EMBEDDING is True
    assert provider.SUPPORTS_LIST_MODELS is True


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    provider = AnyLLM.create("dashscope", api_key="sk-1")
    assert isinstance(provider, DashscopeProvider)
    assert provider.PROVIDER_NAME == "dashscope"
    assert "dashscope" in AnyLLM.get_supported_providers()


def test_dashscope_convert_completion_params_renames_max_tokens() -> None:
    """max_tokens should be remapped to max_completion_tokens by the base provider."""
    params = CompletionParams(
        model_id="qwen-plus",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )
    result = DashscopeProvider._convert_completion_params(params)
    assert result["max_completion_tokens"] == 100
    assert "max_tokens" not in result


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = DashscopeProvider.get_provider_metadata()
    assert metadata.name == "dashscope"
    assert metadata.env_key == "DASHSCOPE_API_KEY"
    assert metadata.env_api_base == "DASHSCOPE_API_BASE"
    assert metadata.doc_url == "https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api"
    assert metadata.completion is True
