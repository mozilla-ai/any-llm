import os

from any_llm.provider import ApiConfig
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
    SUPPORTS_EMBEDDING = False

    def __init__(self, config: ApiConfig) -> None:
        super().__init__(config)
        # Override API_BASE if provided in config or environment
        base = config.api_base or os.getenv("PERPLEXITY_API_BASE")
        if base:
            self.API_BASE = base
