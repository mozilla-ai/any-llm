from google import genai

from any_llm.config import ClientConfig
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.gemini.base import GoogleProvider


class VertexaiProvider(GoogleProvider):
    """Vertex AI Provider using Google Cloud Vertex AI."""

    PROVIDER_NAME = "vertexai"
    PROVIDER_DOCUMENTATION_URL = "https://cloud.google.com/vertex-ai/docs"
    ENV_API_KEY_NAME = ""

    def _verify_and_set_api_key(self, config: ClientConfig) -> ClientConfig:
        # api_key is not mandatory in vertexai
        return config

    def _init_client(self) -> None:
        """Get Vertex AI client."""
        self.client = genai.Client(
            vertexai=True,
            **(self.config.client_args if self.config.client_args else {}),
        )
        if self.client._api_client.project is None:
            msg = "vertexai"
            raise MissingApiKeyError(msg, "GOOGLE_CLOUD_PROJECT")
        if self.client._api_client.location is None:
            msg = "vertexai"
            raise MissingApiKeyError(msg, "GOOGLE_CLOUD_LOCATION")
