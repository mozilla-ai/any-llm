import os

from google import genai

from any_llm.config import ClientConfig
from any_llm.exceptions import MissingApiKeyError

from .base import GoogleProvider


class GeminiProvider(GoogleProvider):
    """Gemini Provider using the Google GenAI Developer API."""

    PROVIDER_NAME = "gemini"
    PROVIDER_DOCUMENTATION_URL = "https://ai.google.dev/gemini-api/docs"
    ENV_API_KEY_NAME = "GEMINI_API_KEY/GOOGLE_API_KEY"

    def _verify_and_set_api_key(self, config: ClientConfig) -> ClientConfig:
        if not config.api_key:
            config.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not config.api_key:
            raise MissingApiKeyError(self.PROVIDER_NAME, self.ENV_API_KEY_NAME)
        return config

    def _get_client(self, config: ClientConfig) -> "genai.Client":
        """Get Gemini API client."""
        return genai.Client(api_key=config.api_key, **(config.client_args if config.client_args else {}))
