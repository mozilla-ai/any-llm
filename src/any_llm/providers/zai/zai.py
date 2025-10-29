from any_llm.providers.openai.base import BaseOpenAIProvider


class ZaiProvider(BaseOpenAIProvider):
    """
    Provider for z.ai API.

    z.ai is an OpenAI-compatible API that provides access to various AI models.
    """

    PROVIDER_NAME = "zai"
    API_BASE = "https://api.z.ai/api/paas/v4/"
    PROVIDER_DOCUMENTATION_URL = "https://docs.z.ai/guides/develop/python/introduction"
    ENV_API_KEY_NAME = "ZAI_API_KEY"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    PACKAGES_INSTALLED = True
