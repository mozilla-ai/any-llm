from any_llm.providers.openai.base import BaseOpenAIProvider


class LlamacppProvider(BaseOpenAIProvider):
    API_BASE = "http://127.0.0.1:8080/v1"
    ENV_API_KEY_NAME = "LLAMA_API_KEY"
    PROVIDER_NAME = "llamacpp"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/ggml-org/llama.cpp"

    SUPPORTS_EMBEDDING = True
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_STREAMING = False
