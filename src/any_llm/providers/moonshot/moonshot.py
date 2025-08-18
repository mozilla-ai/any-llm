from any_llm.providers.openai.base import BaseOpenAIProvider


class MoonshotProvider(BaseOpenAIProvider):
    API_BASE = "https://api.moonshot.ai/v1"
    ENV_API_KEY_NAME = "MOONSHOT_API_KEY"
    PROVIDER_NAME = "moonshot"
    PROVIDER_DOCUMENTATION_URL = "https://platform.moonshot.ai/"

    SUPPORTS_EMBEDDING = False  # Moonshot doesn't host an embedding model
