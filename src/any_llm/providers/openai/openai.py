from .base import BaseOpenAIProvider


class OpenaiProvider(BaseOpenAIProvider):
    API_BASE = "https://api.openai.com/v1"
    ENV_API_KEY_NAME = "OPENAI_API_KEY"
    PROVIDER_NAME = "OpenAI"
    PROVIDER_DOCUMENTATION_URL = "https://platform.openai.com/docs/api-reference"
