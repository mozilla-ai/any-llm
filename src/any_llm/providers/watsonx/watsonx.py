from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from any_llm.provider import Provider

MISSING_PACKAGES_ERROR = None
try:
    from ibm_watsonx_ai import APIClient as WatsonxClient
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    from .utils import (
        _convert_models_list,
        _convert_pydantic_to_watsonx_json,
        _convert_response,
        _convert_streaming_chunk,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from ibm_watsonx_ai import APIClient as WatsonxClient  # noqa: TC004

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
    from any_llm.types.model import Model


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
    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    async def _stream_completion_async(
        self,
        model_inference: ModelInference,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        response_stream = await model_inference.achat_stream(
            messages=messages,
            params=kwargs,
        )
        async for chunk in response_stream:
            yield _convert_streaming_chunk(chunk)

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

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
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
            return self._stream_completion_async(model_inference, params.messages, **kwargs)

        response = await model_inference.achat(
            messages=params.messages,
            params=params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format", "stream"}),
        )

        return _convert_response(response)

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

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        client = WatsonxClient(
            url=self.config.api_base or os.getenv("WATSONX_SERVICE_URL"),
            credentials=Credentials(
                api_key=self.config.api_key, url=self.config.api_base or os.getenv("WATSONX_SERVICE_URL")
            ),
        )
        models_response = client.foundation_models.get_model_specs(**kwargs)

        models_data: dict[str, Any]
        if models_response is None:
            models_data = {"resources": []}
        elif hasattr(models_response, "__iter__") and not isinstance(models_response, dict):
            models_list = list(models_response)
            models_data = {"resources": models_list}
        elif isinstance(models_response, dict):
            models_data = models_response
        else:
            models_data = {"resources": [models_response]}

        return _convert_models_list(models_data)
