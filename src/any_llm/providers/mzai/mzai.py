from any_llm.providers.openai import BaseOpenAIProvider


class MzaiProvider(BaseOpenAIProvider):
    API_BASE = "https://platform-api.any-llm.ai/api/v1"
    ENV_API_KEY_NAME = "ANY_LLM_KEY"
    ENV_API_BASE_NAME = "ANY_LLM_PLATFORM_URL"

    PROVIDER_NAME = "mzai"
    PROVIDER_DOCUMENTATION_URL = "https://any-llm.ai"
    SUPPORTS_RESPONSES = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_BATCH = False
    SUPPORTS_COMPLETION_STREAMING = False
    SUPPORTS_COMPLETION = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
