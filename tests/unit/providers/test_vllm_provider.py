from any_llm.providers.vllm.vllm import VllmProvider


def test_provider_without_api_key() -> None:
    provider = VllmProvider()
    assert provider.PROVIDER_NAME == "vllm"
    assert provider.API_BASE == "http://localhost:8000/v1"
    assert provider.ENV_API_KEY_NAME == "VLLM_API_KEY"


def test_provider_with_api_key() -> None:
    provider = VllmProvider(api_key="test-api-key")
    assert provider.PROVIDER_NAME == "vllm"
