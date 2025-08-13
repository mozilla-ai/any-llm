from any_llm.providers.openai.base import BaseOpenAIProvider


class InceptionProvider(BaseOpenAIProvider):
    API_BASE = "https://api.inceptionlabs.ai/v1"
    ENV_API_KEY_NAME = "INCEPTION_API_KEY"
    PROVIDER_NAME = "inception"
    PROVIDER_DOCUMENTATION_URL = "https://inceptionlabs.ai/"

    SUPPORTS_EMBEDDING = False  # Inception doesn't host an embedding model
