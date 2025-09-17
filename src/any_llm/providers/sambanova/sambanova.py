from any_llm.providers.openai.base import BaseOpenAIProvider


class SambanovaProvider(BaseOpenAIProvider):
    API_BASE = "https://api.sambanova.ai/v1/"
    ENV_API_KEY_NAME = "SAMBANOVA_API_KEY"
    PROVIDER_NAME = "sambanova"
    PROVIDER_DOCUMENTATION_URL = "https://sambanova.ai/"

    SUPPORTS_COMPLETION_PDF = False
