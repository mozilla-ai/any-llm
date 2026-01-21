from collections.abc import AsyncIterator
from typing import Any

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.openresponses_generated import ResponseResource
from any_llm.types.responses import Response, ResponsesParams, ResponseStreamEvent

from .utils import extract_reasoning_from_response


class FireworksProvider(BaseOpenAIProvider):
    PROVIDER_NAME = "fireworks"
    ENV_API_KEY_NAME = "FIREWORKS_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://fireworks.ai/api"
    API_BASE = "https://api.fireworks.ai/inference/v1"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    async def _aresponses(
        self, params: ResponsesParams, **kwargs: Any
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """Call Fireworks Responses API and extract reasoning from think tags."""
        response = await super()._aresponses(params, **kwargs)
        if isinstance(response, ResponseResource):
            return extract_reasoning_from_response(response)
        return response
