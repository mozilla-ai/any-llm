from any_llm.providers.openai.base import BaseOpenAIProvider


class XaiProvider(BaseOpenAIProvider):
    API_BASE = "https://api.x.ai/v1"
    ENV_API_KEY_NAME = "XAI_API_KEY"
    PROVIDER_NAME = "xAI"
    PROVIDER_DOCUMENTATION_URL = "https://x.ai/"

    SUPPORTS_EMBEDDING = False
