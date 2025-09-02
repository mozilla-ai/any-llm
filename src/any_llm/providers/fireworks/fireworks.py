from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncStream

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import Reasoning
from any_llm.types.responses import Response, ResponseStreamEvent


class FireworksProvider(BaseOpenAIProvider):
    PROVIDER_NAME = "fireworks"
    ENV_API_KEY_NAME = "FIREWORKS_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://fireworks.ai/api"
    API_BASE = "https://api.fireworks.ai/inference/v1"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    async def aresponses(
        self, model: str, input_data: Any, **kwargs: Any
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """Call Fireworks Responses API and normalize into ChatCompletion/Chunks."""
        response = await super().aresponses(model, input_data, **kwargs)

        if isinstance(response, Response) and not isinstance(response, AsyncStream):
            # See https://fireworks.ai/blog/response-api for details about Fireworks Responses API support
            reasoning = response.output[-1].content[0].text.split("</think>")[-1]  # type: ignore[union-attr,index]
            if reasoning:
                reasoning = reasoning.strip()
                response.output[-1].content[0].text = response.output[-1].content[0].text.split("</think>")[0]  # type: ignore[union-attr,index]
            response.reasoning = Reasoning(content=reasoning) if reasoning else None  # type: ignore[assignment]

        return response
