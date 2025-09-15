import os
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
    ENV_API_KEY_NAME = "GOOGLE_PROJECT_ID"

    def _get_client(self, config: ClientConfig) -> "genai.Client":
        """Get Vertex AI client."""
        from google import genai

        project_id = os.getenv("GOOGLE_PROJECT_ID")
        location = os.getenv("GOOGLE_REGION", "us-central1")

        if not project_id:
            msg = "Google Vertex AI"
            raise MissingApiKeyError(msg, "GOOGLE_PROJECT_ID")

        return genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            **(config.client_args if config.client_args else {}),
        )
