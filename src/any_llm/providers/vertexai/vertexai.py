from typing import TYPE_CHECKING

from any_llm.config import ClientConfig
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.gemini.base import GoogleProvider

if TYPE_CHECKING:
    from google import genai


class VertexaiProvider(GoogleProvider):
    """Vertex AI Provider using Google Cloud Vertex AI."""

    PROVIDER_NAME = "vertexai"
    PROVIDER_DOCUMENTATION_URL = "https://cloud.google.com/vertex-ai/docs"
    ENV_API_KEY_NAME = ""

    def _verify_and_set_api_key(self, config: ClientConfig) -> ClientConfig:
        # api_key is not mandatory in vertexai
        return config

    def _get_client(self, config: ClientConfig) -> "genai.Client":
        """Get Vertex AI client."""
        from google import genai

        client = genai.Client(
            vertexai=True,
            api_key=self.config.api_key,
            **(config.client_args if config.client_args else {}),
        )
        if client._api_client.project is None:
            msg = "vertexai"
            raise MissingApiKeyError(msg, "GOOGLE_CLOUD_PROJECT")
        if client._api_client.location is None:
            msg = "vertexai"
            raise MissingApiKeyError(msg, "GOOGLE_CLOUD_LOCATION")

        return client
