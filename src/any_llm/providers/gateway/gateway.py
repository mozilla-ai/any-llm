import os
from typing import Any

from any_llm.logging import logger
from any_llm.providers.openai.base import BaseOpenAIProvider

GATEWAY_HEADER_NAME = "X-AnyLLM-Key"


class GatewayProvider(BaseOpenAIProvider):
    ENV_API_KEY_NAME = "GATEWAY_API_KEY"
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
            msg = "For any-llm-gateway, api_base is required"
            raise ValueError(msg)
        api_key = self._verify_and_set_api_key(api_key)
        if api_key:
            if not kwargs:
                # if passed into client init, the parameters
                kwargs = {"default_headers": {}}
            elif kwargs.get("default_headers", {}).get(GATEWAY_HEADER_NAME):
                msg = f"{GATEWAY_HEADER_NAME} header is already set, overriding with new API key"
                logger.info(msg)
            kwargs["default_headers"][GATEWAY_HEADER_NAME] = f"Bearer {api_key}"
        super().__init__(api_key=api_key, api_base=api_base, **kwargs)

    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        """Unlike other providers, the gateway provider does not require an API key"""
        return api_key or os.getenv(self.ENV_API_KEY_NAME, "")
