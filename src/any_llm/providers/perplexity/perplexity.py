from any_llm.providers.openai.base import BaseOpenAIProvider


class PerplexityProvider(BaseOpenAIProvider):
    """Perplexity provider for accessing LLMs through Perplexity's OpenAI-compatible API."""

    PACKAGES_INSTALLED = True

    API_BASE = "https://api.perplexity.ai"
    ENV_API_KEY_NAME = "PERPLEXITY_API_KEY"
    PROVIDER_NAME = "perplexity"
    PROVIDER_DOCUMENTATION_URL = "https://docs.perplexity.ai/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = False
