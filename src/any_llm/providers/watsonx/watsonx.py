import os
from collections.abc import Iterator
from typing import Any

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from pydantic import BaseModel

from any_llm.provider import Provider
from any_llm.providers.watsonx.utils import (
    _convert_pydantic_to_watsonx_json,
    _convert_response,
    _convert_streaming_chunk,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class WatsonxProvider(Provider):
    """IBM Watsonx Provider using the official IBM Watsonx AI SDK."""

    PROVIDER_NAME = "watsonx"
    ENV_API_KEY_NAME = "WATSONX_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://www.ibm.com/watsonx"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = False

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    def _stream_completion(
        self,
        model_inference: ModelInference,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        response_stream = model_inference.chat_stream(
            messages=messages,
            params=kwargs,
        )
        for chunk in response_stream:
            yield _convert_streaming_chunk(chunk)

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Watsonx."""

        model_inference = ModelInference(
            model_id=params.model_id,
            credentials=Credentials(
                api_key=self.config.api_key,
                url=self.config.api_base or os.getenv("WATSONX_SERVICE_URL"),
            ),
            project_id=kwargs.get("project_id") or os.getenv("WATSONX_PROJECT_ID"),
        )

        # Handle response_format by inlining schema guidance into the prompt
        response_format = params.response_format
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            params.messages = _convert_pydantic_to_watsonx_json(response_format, params.messages)

        if params.reasoning_effort == "auto":
            params.reasoning_effort = None

        if params.stream:
            kwargs = {
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format", "stream"}),
                **kwargs,
            }
            return self._stream_completion(model_inference, params.messages, **kwargs)

        response = model_inference.chat(
            messages=params.messages,
            params=params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format", "stream"}),
        )

        return _convert_response(response)
