from any_llm.providers.openai.base import BaseOpenAIProvider


class VllmProvider(BaseOpenAIProvider):
    API_BASE = "http://localhost:8000/v1"
    ENV_API_KEY_NAME = "VLLM_API_KEY"
    PROVIDER_NAME = "vllm"
    PROVIDER_DOCUMENTATION_URL = "https://docs.vllm.ai/"

    SUPPORTS_EMBEDDING = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION_PDF = False

    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # vLLM server by default doesn't require an API key
        # but can be configured to use one via --api-key flag
        return api_key or ""
