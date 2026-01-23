from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from openai._streaming import AsyncStream
from openai.types.responses import Response as OpenAIResponse

from any_llm.providers.openai.base import BaseOpenAIProvider

if TYPE_CHECKING:
    from openai.types.responses import ResponseStreamEvent as OpenAIResponseStreamEvent
    from openresponses_types import CreateResponseBody, ResponseResource


class FireworksProvider(BaseOpenAIProvider):
    PROVIDER_NAME = "fireworks"
    ENV_API_KEY_NAME = "FIREWORKS_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://fireworks.ai/api"
    API_BASE = "https://api.fireworks.ai/inference/v1"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    SUPPORTS_RESPONSES = True

    async def _aresponses(
        self, params: "CreateResponseBody", **kwargs: Any
    ) -> "ResponseResource | OpenAIResponse | AsyncIterator[dict[str, Any]]":
        """Call Fireworks Responses API and return OpenAI Response type directly.

        Fireworks is not yet compliant with the OpenResponses spec, so we return
        the OpenAI Response type directly instead of converting to ResponseResource.
        """
        response: OpenAIResponse | AsyncStream[OpenAIResponseStreamEvent] = await self.client.responses.create(
            **params.model_dump(exclude_none=True), **kwargs
        )

        if isinstance(response, OpenAIResponse):
            return response

        if isinstance(response, AsyncStream):

            async def stream_iterator() -> AsyncIterator[dict[str, Any]]:
                async for event in response:
                    yield event.model_dump()

            return stream_iterator()

        msg = f"Responses API returned an unexpected type: {type(response)}"
        raise ValueError(msg)
