from any_llm.providers.openai.base import BaseOpenAIProvider


class DeepinfraProvider(BaseOpenAIProvider):
    API_BASE = "https://api.deepinfra.com/v1/openai"
    ENV_API_KEY_NAME = "DEEPINFRA_API_KEY"
    ENV_API_BASE_NAME = "DEEPINFRA_API_BASE"
    PROVIDER_NAME = "deepinfra"
    PROVIDER_DOCUMENTATION_URL = "https://deepinfra.com/docs/openai_api"

    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_PDF = False
