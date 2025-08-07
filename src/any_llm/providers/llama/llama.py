from any_llm.providers.openai.base import BaseOpenAIProvider


class LlamaProvider(BaseOpenAIProvider):
    """Llama provider for accessing multiple LLMs through Llama's API."""

    API_BASE = "https://api.llama.com/compat/v1/"
    ENV_API_KEY_NAME = "LLAMA_API_KEY"
    PROVIDER_NAME = "Llama API"
    PROVIDER_DOCUMENTATION_URL = "https://www.llama.com/products/llama-api/"

    SUPPORTS_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_REASONING = False
    SUPPORTS_EMBEDDING = False
