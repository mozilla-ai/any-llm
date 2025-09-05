import os

from google import genai

from any_llm.exceptions import MissingApiKeyError
from any_llm.provider import ClientConfig

from .base import GoogleProvider


class GeminiProvider(GoogleProvider):
    """Gemini Provider using the Google GenAI Developer API."""

    PROVIDER_NAME = "gemini"
    PROVIDER_DOCUMENTATION_URL = "https://ai.google.dev/gemini-api/docs"
    ENV_API_KEY_NAME = "GEMINI_API_KEY/GOOGLE_API_KEY"

    def _get_client(self, config: ClientConfig) -> "genai.Client":
        """Get Gemini API client."""
        api_key = getattr(config, "api_key", None) or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            msg = "Google Gemini Developer API"
            raise MissingApiKeyError(msg, "GEMINI_API_KEY/GOOGLE_API_KEY")

        return genai.Client(api_key=api_key, **(config.client_args if config.client_args else {}))
