from typing import Any

from google import genai
from typing_extensions import override

from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.gemini.base import GoogleProvider


class VertexaiProvider(GoogleProvider):
    """Vertex AI Provider using Google Cloud Vertex AI."""

    PROVIDER_NAME = "vertexai"
    PROVIDER_DOCUMENTATION_URL = "https://cloud.google.com/vertex-ai/docs"
    ENV_API_KEY_NAME = ""
    ENV_API_BASE_NAME = "VERTEXAI_API_BASE"

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        return api_key

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        """Get Vertex AI client."""

        # Ensure timeout is correctly configured if present.
        if (timeout := kwargs.pop("timeout", None)) is not None:
            GoogleProvider._merge_timeout_into_http_options(timeout, kwargs)

        if ('service_account' in kwargs
                and 'project' in kwargs
                and 'location' in kwargs):

            credentials = self._build_credentials(kwargs['service_account'])
            project = kwargs['project']
            location = kwargs['location']

            exclude = {"service_account", "project", "location"}
            kwargs_copy = {k: v for k, v in kwargs.items() if k not in exclude}

            self.client = genai.client.Client(
                vertexai=True,
                credentials=credentials,
                project=project,
                location=location,
                **kwargs_copy,
            )
        else:
            self.client = genai.client.Client(
                vertexai=True,
                **kwargs,
            )

        if self.client._api_client.project is None:
            msg = "vertexai"
            raise MissingApiKeyError(msg, "GOOGLE_CLOUD_PROJECT")
        if self.client._api_client.location is None:
            msg = "vertexai"
            raise MissingApiKeyError(msg, "GOOGLE_CLOUD_LOCATION")
