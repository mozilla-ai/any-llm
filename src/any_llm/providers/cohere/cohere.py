from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import UnsupportedParameterError

MISSING_PACKAGES_ERROR = None
try:
    import cohere
    from cohere import V2ChatResponse

    from .utils import (
        _convert_models_list,
        _convert_response,
        _create_openai_chunk_from_cohere_chunk,
        _patch_messages,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


class CohereProvider(AnyLLM):
    """Cohere Provider using the new response conversion utilities."""

    PROVIDER_NAME = "cohere"
    ENV_API_KEY_NAME = "COHERE_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://cohere.com/api"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: cohere.AsyncClientV2

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Cohere API."""
        # Cohere does not support providing reasoning effort
        converted_params = params.model_dump(
            exclude_none=True, exclude={"model_id", "messages", "response_format", "stream"}
        )
        if converted_params.get("reasoning_effort") == "auto":
            converted_params.pop("reasoning_effort")
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_completion_response(response: V2ChatResponse, **kwargs: Any) -> ChatCompletion:
        """Convert Cohere response to OpenAI format."""
        # We need the model parameter for conversion
        model = kwargs.get("model", "unknown")
        return _convert_response(response, model)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Cohere chunk response to OpenAI format."""
        return _create_openai_chunk_from_cohere_chunk(response)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Cohere."""
        msg = "Cohere does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Cohere embedding response to OpenAI format."""
        msg = "Cohere does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Cohere list models response to OpenAI format."""
        return _convert_models_list(response)

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = cohere.AsyncClientV2(api_key=api_key, **kwargs)

    async def _stream_completion_async(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        cohere_stream = self.client.chat_stream(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        async for chunk in cohere_stream:
            yield self._convert_completion_chunk_response(chunk)

    @staticmethod
    def _preprocess_response_format(response_format: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
        # if response format is a BaseModel, generate model json schema
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return {"type": "json_object", "schema": response_format.model_json_schema()}
        # can either be json schema already in dict
        # or {"type": "json_object"} to just generate *a* JSON (JSON mode)
        # see docs here: https://docs.cohere.com/docs/structured-outputs#json-mode
        if isinstance(response_format, dict):
            return response_format
        # For now, let Cohere API handle invalid schemas.
        # Note that Cohere has a bunch of limitations on JSON schemas (e.g., no oneOf, numeric/str ranges, weird regex limitations)
        # see docs here: https://docs.cohere.com/docs/structured-outputs#unsupported-schema-features
        # Validation logic could/would eventually go here
        return response_format

    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if params.response_format is not None:
            kwargs["response_format"] = self._preprocess_response_format(params.response_format)
        if params.stream and params.response_format is not None:
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        if params.parallel_tool_calls is not None:
            msg = "parallel_tool_calls"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)

        completion_kwargs = self._convert_completion_params(params, **kwargs)
        patched_messages = _patch_messages(params.messages)

        if params.stream:
            return self._stream_completion_async(
                params.model_id,
                patched_messages,
                **completion_kwargs,
            )

        response = await self.client.chat(
            model=params.model_id,
            messages=patched_messages,  # type: ignore[arg-type]
            **completion_kwargs,
        )

        return self._convert_completion_response(response, model=params.model_id)

    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        model_list = await self.client.models.list(**kwargs)
        return self._convert_list_models_response(model_list)
