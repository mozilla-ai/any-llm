from any_llm.providers.openai.base import BaseOpenAIProvider


class NebiusProvider(BaseOpenAIProvider):
    DEFAULT_API_BASE = "https://api.studio.nebius.ai/v1"
    ENV_API_KEY_NAME = "NEBIUS_API_KEY"
    PROVIDER_NAME = "Nebius"
    PROVIDER_DOCUMENTATION_URL = "https://studio.nebius.ai/"
