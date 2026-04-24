from any_llm.providers.openai.base import BaseOpenAIProvider


class ZyphraProvider(BaseOpenAIProvider):
    API_BASE = "https://uyppidoc.zyphracloud.com/v1"
    ENV_API_KEY_NAME = "ZYPHRA_API_KEY"
    ENV_API_BASE_NAME = "ZYPHRA_API_BASE"
    PROVIDER_NAME = "zyphra"
    PROVIDER_DOCUMENTATION_URL = "https://zyphra.com"

    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_LIST_MODELS = False
