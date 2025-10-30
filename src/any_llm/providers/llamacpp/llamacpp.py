from any_llm.providers.openai.base import BaseOpenAIProvider


class LlamacppProvider(BaseOpenAIProvider):
    API_BASE = "http://127.0.0.1:8080/v1"
    ENV_API_KEY_NAME = "None"
    PROVIDER_NAME = "llamacpp"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/ggml-org/llama.cpp"

    SUPPORTS_EMBEDDING = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION_PDF = False

    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        return ""
