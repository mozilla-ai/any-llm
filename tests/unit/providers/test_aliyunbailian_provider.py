import pytest

from any_llm.providers.aliyunbailian.aliyunbailian import AliyunbailianProvider
from any_llm.types.completion import CompletionParams


def test_aliyunbailian_provider_initialization() -> None:
    """Test that AliyunbailianProvider can be instantiated with an API key."""
    # Should not raise MissingApiKeyError because we provide a dummy key
    provider = AliyunbailianProvider(api_key="dummy_key")
    assert provider.PROVIDER_NAME == "aliyunbailian"
    assert provider.ENV_API_KEY_NAME == "ALIYUNBAILIAN_API_KEY"
    assert provider.ENV_API_BASE_NAME == "ALIYUNBAILIAN_API_BASE"
    assert provider.API_BASE == "https://bailian.aliyun.com/v1/"


def test_aliyunbailian_supports_flags() -> None:
    """Test that AliyunbailianProvider has correct feature support flags."""
    provider = AliyunbailianProvider(api_key="dummy_key")
    assert provider.SUPPORTS_COMPLETION_IMAGE is False
    assert provider.SUPPORTS_COMPLETION_PDF is False
    assert provider.SUPPORTS_EMBEDDING is False
    assert provider.SUPPORTS_COMPLETION_REASONING is False
    # Default flags from BaseOpenAIProvider
    assert provider.SUPPORTS_COMPLETION_STREAMING is True
    assert provider.SUPPORTS_COMPLETION is True
    assert provider.SUPPORTS_LIST_MODELS is True


def test_aliyunbailian_convert_completion_params_passthrough() -> None:
    """Test that _convert_completion_params passes through without modification."""
    params = CompletionParams(
        model_id="aliyunbailian-model",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )
    result = AliyunbailianProvider._convert_completion_params(params)
    # Should have max_completion_tokens renamed from max_tokens
    assert "max_completion_tokens" in result
    assert result["max_completion_tokens"] == 100
    # Should not have max_tokens
    assert "max_tokens" not in result