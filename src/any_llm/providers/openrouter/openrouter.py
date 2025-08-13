from any_llm.providers.openai.base import BaseOpenAIProvider


class OpenrouterProvider(BaseOpenAIProvider):
    """OpenRouter provider for accessing multiple LLMs through OpenRouter's API."""

    API_BASE = "https://openrouter.ai/api/v1"
    ENV_API_KEY_NAME = "OPENROUTER_API_KEY"
    PROVIDER_NAME = "openrouter"
    PROVIDER_DOCUMENTATION_URL = "https://openrouter.ai/docs"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False
