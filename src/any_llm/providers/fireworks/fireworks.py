from any_llm.providers.openai.base import BaseOpenAIProvider


class FireworksProvider(BaseOpenAIProvider):
    PROVIDER_NAME = "fireworks"
    ENV_API_KEY_NAME = "FIREWORKS_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://fireworks.ai/api"
    BASE_URL = "https://api.fireworks.ai/inference/v1"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
