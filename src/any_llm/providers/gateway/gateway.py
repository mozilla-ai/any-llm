from typing import Any

from any_llm.providers.openai.base import BaseOpenAIProvider


class GatewayProvider(BaseOpenAIProvider):
    ENV_API_KEY_NAME = "None"
    PROVIDER_NAME = "gateway"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/mozilla-ai/any-llm"

    # All features are marked as supported, but depending on which provider you call inside the gateway, they may not all work.
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = True

    def __init__(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        if not api_base:
            msg = "For any-llm gateway, api_base is required"
            raise ValueError(msg)
        super().__init__(api_key=api_key, api_base=api_base, **kwargs)
