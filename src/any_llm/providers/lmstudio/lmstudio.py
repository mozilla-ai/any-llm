import os

from any_llm.config import ClientConfig
from any_llm.providers.openai.base import BaseOpenAIProvider

# LM Studio has a python sdk, but per their docs they are compliant with OpenAI spec
# https://lmstudio.ai/docs/app/api/endpoints/openai
# So until its clear why the python sdk should be used, we'll default to inheriting from OpenAI SDK.


class LmstudioProvider(BaseOpenAIProvider):
    API_BASE = "http://localhost:1234/v1"
    ENV_API_KEY_NAME = "LM_STUDIO_API_KEY"
    PROVIDER_NAME = "lmstudio"
    PROVIDER_DOCUMENTATION_URL = "https://lmstudio.ai/"

    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_LIST_MODELS = True

    def __init__(self, config: ClientConfig) -> None:
        """We don't use the Provider init because by default we don't require an API key."""

        self.url = config.api_base or os.getenv("LMSTUDIO_API_URL")
        self.config = config
        self.config.api_key = ""  # In order to be compatible with the OpenAI client, the API key cannot be None if the OPENAI_API_KEY environment variable is not set (which is the case for LMStudio)
