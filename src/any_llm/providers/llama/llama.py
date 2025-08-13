from any_llm.providers.openai.base import BaseOpenAIProvider


class LlamaProvider(BaseOpenAIProvider):
    """Llama provider for accessing multiple LLMs through Llama's API."""

    API_BASE = "https://api.llama.com/compat/v1/"
    ENV_API_KEY_NAME = "LLAMA_API_KEY"
    PROVIDER_NAME = "llama"
    PROVIDER_DOCUMENTATION_URL = "https://www.llama.com/products/llama-api/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False
