from any_llm.providers.openai.base import BaseOpenAIProvider


class PinstripesProvider(BaseOpenAIProvider):
    API_BASE = "https://api.pinstripes.io/v1"
    ENV_API_KEY_NAME = "PINSTRIPES_API_KEY"
    ENV_API_BASE_NAME = "PINSTRIPES_API_BASE"
    PROVIDER_NAME = "pinstripes"
    PROVIDER_DOCUMENTATION_URL = "https://pinstripes.io"

    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
