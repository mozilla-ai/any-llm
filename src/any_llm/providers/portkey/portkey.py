from any_llm.providers.openai.base import BaseOpenAIProvider

try:
    from portkey_ai import PORTKEY_GATEWAY_URL
except ImportError:
    msg = "portkey is not installed. Please install it with `pip install any-llm-sdk[portkey]`"
    raise ImportError(msg)


class PortkeyProvider(BaseOpenAIProvider):
    """Portkey provider for accessing 200+ LLMs through Portkey's AI Gateway."""

    API_BASE = PORTKEY_GATEWAY_URL
    ENV_API_KEY_NAME = "PORTKEY_API_KEY"
    PROVIDER_NAME = "Portkey"
    PROVIDER_DOCUMENTATION_URL = "https://portkey.ai/docs"

    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDING = False
