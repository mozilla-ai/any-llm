from any_llm.providers.openai.base import BaseOpenAIProvider


class MoonshotProvider(BaseOpenAIProvider):
    DEFAULT_API_BASE = "https://api.moonshot.ai/v1"
    ENV_API_KEY_NAME = "MOONSHOT_API_KEY"
    PROVIDER_NAME = "Moonshot AI"
    PROVIDER_DOCUMENTATION_URL = "https://platform.moonshot.ai/"
