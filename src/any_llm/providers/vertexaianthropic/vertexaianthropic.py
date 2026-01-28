from __future__ import annotations

import os
from typing import Any

from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.anthropic.base import BaseAnthropicProvider

MISSING_PACKAGES_ERROR = None
try:
    from anthropic import AsyncAnthropicVertex
except ImportError as e:
    MISSING_PACKAGES_ERROR = e


class VertexaianthropicProvider(BaseAnthropicProvider):
    """
    VertexAI Anthropic Provider for Claude models through Google Cloud's Model Garden.

    Uses Anthropic's `AsyncAnthropicVertex` client from `anthropic[vertex]` to access
    Claude models deployed on Google Cloud VertexAI.

    Authentication is handled via Google Cloud Application Default Credentials (ADC),
    not an API key. Requires `GOOGLE_CLOUD_PROJECT` environment variable or `project_id`
    constructor argument.
    """

    PROVIDER_NAME = "vertexaianthropic"
    ENV_API_KEY_NAME = ""  # VertexAI uses GCP ADC, not an API key
    PROVIDER_DOCUMENTATION_URL = "https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude"

    SUPPORTS_LIST_MODELS = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncAnthropicVertex

    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # VertexAI uses Google Cloud ADC, not an API key
        # We don't require an API key, but we do require project_id
        return api_key

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        project_id = kwargs.pop("project_id", None) or os.getenv("GOOGLE_CLOUD_PROJECT")
        region = kwargs.pop("region", None) or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not project_id:
            raise MissingApiKeyError(self.PROVIDER_NAME, "GOOGLE_CLOUD_PROJECT")

        self.client = AsyncAnthropicVertex(
            project_id=project_id,
            region=region,
            **kwargs,
        )
