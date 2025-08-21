from any_llm.providers.openai.base import BaseOpenAIProvider


class PortkeyProvider(BaseOpenAIProvider):
    """Portkey provider for accessing 200+ LLMs through Portkey's AI Gateway."""

    API_BASE = "https://api.portkey.ai/v1"
    ENV_API_KEY_NAME = "PORTKEY_API_KEY"
    PROVIDER_NAME = "portkey"
    PROVIDER_DOCUMENTATION_URL = "https://portkey.ai/docs"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    _DEFAULT_REASONING_EFFORT = None
