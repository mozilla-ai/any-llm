from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from any_llm.any_llm import AnyLLM

MISSING_PACKAGES_ERROR = None
try:
    from ibm_watsonx_ai import APIClient as WatsonxClient
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models.inference.model_inference import ModelInference

    from .utils import (
        _convert_models_list,
        _convert_pydantic_to_watsonx_json,
        _convert_response,
        _convert_streaming_chunk,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from ibm_watsonx_ai import APIClient as WatsonxClient  # noqa: TC004

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


class WatsonxProvider(AnyLLM):
    """IBM Watsonx Provider using the official IBM Watsonx AI SDK."""

    PROVIDER_NAME = "watsonx"
    ENV_API_KEY_NAME = "WATSONX_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://www.ibm.com/watsonx"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Watsonx API."""
        # Watsonx does not support providing reasoning effort
        converted_params = params.model_dump(
            exclude_none=True, exclude={"model_id", "messages", "response_format", "stream"}
        )
        if converted_params.get("reasoning_effort") == "auto":
            converted_params.pop("reasoning_effort")
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Watsonx response to OpenAI format."""
        return _convert_response(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Watsonx chunk response to OpenAI format."""
        return _convert_streaming_chunk(response)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Watsonx."""
        msg = "Watsonx does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Watsonx embedding response to OpenAI format."""
        msg = "Watsonx does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Watsonx list models response to OpenAI format."""
        return _convert_models_list(response)

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        # watsonx requires params.model_id to instantiate the client
        # which is not available at this point.
        self.api_key = api_key
        self.api_base = api_base
        self.kwargs = kwargs

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
            yield self._convert_completion_chunk_response(chunk)

    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using Watsonx."""

        model_inference = ModelInference(
            model_id=params.model_id,
            credentials=Credentials(api_key=self.api_key, url=self.api_base),
            **self.kwargs,
        )

        # Handle response_format by inlining schema guidance into the prompt
        response_format = params.response_format
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            params.messages = _convert_pydantic_to_watsonx_json(response_format, params.messages)

        if params.reasoning_effort == "auto":
            params.reasoning_effort = None

        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return self._stream_completion_async(model_inference, params.messages, **completion_kwargs)

        response = await model_inference.achat(
            messages=params.messages,
            params=completion_kwargs,
        )

        return self._convert_completion_response(response)

    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        client = WatsonxClient(
            url=self.api_base,
            credentials=Credentials(api_key=self.api_key, url=self.api_base),
            **self.kwargs,
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

        return self._convert_list_models_response(models_data)
