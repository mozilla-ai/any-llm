import os
from typing import Any

from google import genai
from google.genai import types
from typing_extensions import override

from any_llm.exceptions import MissingApiKeyError

from .base import GoogleProvider


class GeminiProvider(GoogleProvider):
    """Gemini Provider using the Google GenAI Developer API."""

    PROVIDER_NAME = "gemini"
    PROVIDER_DOCUMENTATION_URL = "https://ai.google.dev/gemini-api/docs"
    ENV_API_KEY_NAME = "GEMINI_API_KEY/GOOGLE_API_KEY"
    ENV_API_BASE_NAME = "GOOGLE_GEMINI_BASE_URL"

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise MissingApiKeyError(self.PROVIDER_NAME, self.ENV_API_KEY_NAME)
        return api_key

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        if api_base:
            http_options = kwargs.pop("http_options", None)
            if http_options is None:
                http_options = types.HttpOptions(base_url=api_base)
            elif isinstance(http_options, dict):
                http_options.setdefault("base_url", api_base)
            elif isinstance(http_options, types.HttpOptions) and http_options.base_url is None:
                http_options.base_url = api_base
            kwargs["http_options"] = http_options
        self.client = genai.Client(api_key=api_key, **kwargs)
