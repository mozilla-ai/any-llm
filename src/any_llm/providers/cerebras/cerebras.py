from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import Provider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.types.model import Model

MISSING_PACKAGES_ERROR = None
try:
    import cerebras.cloud.sdk as cerebras
    from cerebras.cloud.sdk.types.chat.chat_completion import ChatChunkResponse

    from .utils import (
        _convert_models_list,
        _convert_response,
        _create_openai_chunk_from_cerebras_chunk,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e


class CerebrasProvider(Provider):
    """Cerebras Provider using the official Cerebras SDK with instructor support for structured outputs."""

    PROVIDER_NAME = "cerebras"
    ENV_API_KEY_NAME = "CEREBRAS_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.cerebras.ai/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Cerebras API."""
        # Cerebras does not support providing reasoning effort
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages", "stream"})
        if converted_params.get("reasoning_effort") == "auto":
            converted_params.pop("reasoning_effort")
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Cerebras response to OpenAI format."""
        return _convert_response(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Cerebras chunk response to OpenAI format."""
        if isinstance(response, ChatChunkResponse):
            return _create_openai_chunk_from_cerebras_chunk(response)
        msg = f"Unsupported chunk type: {type(response)}"
        raise ValueError(msg)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Cerebras."""
        msg = "Cerebras does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Cerebras embedding response to OpenAI format."""
        msg = "Cerebras does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Cerebras list models response to OpenAI format."""
        return _convert_models_list(response)

    async def _stream_completion_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""

        client = cerebras.AsyncCerebras(
            api_key=self.config.api_key, **(self.config.client_args if self.config.client_args else {})
        )

        if kwargs.get("response_format", None) is not None:
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        cerebras_stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        async for chunk in cast("cerebras.AsyncStream[ChatCompletion]", cerebras_stream):
            yield self._convert_completion_chunk_response(chunk)

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using Cerebras with instructor support for structured outputs."""

        if params.response_format:
            # See https://inference-docs.cerebras.ai/capabilities/structured-outputs for guide to creating schema
            if isinstance(params.response_format, type) and issubclass(params.response_format, BaseModel):
                params.response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": params.response_format.model_json_schema(),
                        "strict": True,
                    },
                }

        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return self._stream_completion_async(
                params.model_id,
                params.messages,
                **completion_kwargs,
            )

        client = cerebras.AsyncCerebras(
            api_key=self.config.api_key, **(self.config.client_args if self.config.client_args else {})
        )

        response = await client.chat.completions.create(
            model=params.model_id,
            messages=params.messages,
            **completion_kwargs,
        )

        if hasattr(response, "model_dump"):
            response_data = response.model_dump()
        else:
            msg = "Streaming responses are not supported in this context"
            raise ValueError(msg)

        return self._convert_completion_response(response_data)

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        client = cerebras.Cerebras(
            api_key=self.config.api_key, **(self.config.client_args if self.config.client_args else {})
        )
        models_list = client.models.list(**kwargs)
        return self._convert_list_models_response(models_list)
