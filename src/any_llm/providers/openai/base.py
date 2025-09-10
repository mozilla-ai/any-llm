import os
from abc import ABC
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, cast

from openai import AsyncOpenAI, OpenAI
from openai._streaming import AsyncStream
from openai._types import NOT_GIVEN
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk

from any_llm.logging import logger
from any_llm.provider import Provider
from any_llm.providers.openai.utils import _convert_chat_completion, _normalize_openai_dict_response
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.types.model import Model
from any_llm.types.responses import Response, ResponseStreamEvent


class BaseOpenAIProvider(Provider, ABC):
    """
    Base provider for OpenAI-compatible services.

    This class provides a common foundation for providers that use OpenAI-compatible APIs.
    Subclasses only need to override configuration defaults and client initialization
    if needed.
    """

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True

    PACKAGES_INSTALLED = True

    _DEFAULT_REASONING_EFFORT: Literal["minimal", "low", "medium", "high", "auto"] | None = None

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for OpenAI API."""
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages"})
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert OpenAI response to OpenAI format (passthrough)."""
        if isinstance(response, OpenAIChatCompletion):
            return _convert_chat_completion(response)
        # If it's already our ChatCompletion type, return it
        if isinstance(response, ChatCompletion):
            return response
        # Otherwise, validate it as our type
        return ChatCompletion.model_validate(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert OpenAI chunk response to OpenAI format (passthrough)."""
        if isinstance(response, OpenAIChatCompletionChunk):
            if not isinstance(response.created, int):
                logger.warning(
                    "API returned an unexpected created type: %s. Setting to int.",
                    type(response.created),
                )
                response.created = int(response.created)
            normalized_chunk = _normalize_openai_dict_response(response.model_dump())
            return ChatCompletionChunk.model_validate(normalized_chunk)
        # If it's already our ChatCompletionChunk type, return it
        if isinstance(response, ChatCompletionChunk):
            return response
        # Otherwise, validate it as our type
        return ChatCompletionChunk.model_validate(response)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for OpenAI API."""
        converted_params = {"input": params}
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert OpenAI embedding response to OpenAI format (passthrough)."""
        if isinstance(response, CreateEmbeddingResponse):
            return response
        return CreateEmbeddingResponse.model_validate(response)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert OpenAI list models response to OpenAI format (passthrough)."""
        if hasattr(response, "data"):
            # Validate each model in the data
            return [Model.model_validate(model) if not isinstance(model, Model) else model for model in response.data]
        # If it's already a sequence of our Model type, return it
        if isinstance(response, (list, tuple)) and all(isinstance(item, Model) for item in response):
            return response
        # Otherwise, validate each item
        return [Model.model_validate(item) if not isinstance(item, Model) else item for item in response]

    def _get_client(self, sync: bool = False) -> AsyncOpenAI | OpenAI:
        _client_class = OpenAI if sync else AsyncOpenAI
        return _client_class(
            base_url=self.config.api_base or self.API_BASE or os.getenv("OPENAI_API_BASE"),
            api_key=self.config.api_key,
            **(self.config.client_args if self.config.client_args else {}),
        )

    def _convert_completion_response_async(
        self, response: OpenAIChatCompletion | AsyncStream[OpenAIChatCompletionChunk]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Convert an OpenAI completion response to an AnyLLM completion response."""
        if isinstance(response, OpenAIChatCompletion):
            return self._convert_completion_response(response)

        async def chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in response:
                yield self._convert_completion_chunk_response(chunk)

        return chunk_iterator()

    async def acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        client = cast("AsyncOpenAI", self._get_client(sync=False))

        if params.reasoning_effort == "auto":
            params.reasoning_effort = self._DEFAULT_REASONING_EFFORT

        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.response_format:
            if params.stream:
                msg = "stream is not supported for response_format"
                raise ValueError(msg)
            completion_kwargs.pop("stream", None)
            response = await client.chat.completions.parse(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **completion_kwargs,
            )
        else:
            response = await client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **completion_kwargs,
            )
        return self._convert_completion_response_async(response)

    async def aresponses(
        self, model: str, input_data: Any, **kwargs: Any
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """Call OpenAI Responses API"""
        client = self._get_client()

        response = await client.responses.create(
            model=model,
            input=input_data,
            **kwargs,
        )
        if not isinstance(response, Response | AsyncStream):
            msg = f"Responses API returned an unexpected type: {type(response)}"
            raise ValueError(msg)
        return response

    async def aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        # Classes that inherit from BaseOpenAIProvider may override SUPPORTS_EMBEDDING
        if not self.SUPPORTS_EMBEDDING:
            msg = "This provider does not support embeddings."
            raise NotImplementedError(msg)

        client = cast("AsyncOpenAI", self._get_client())

        embedding_kwargs = self._convert_embedding_params(inputs, **kwargs)
        return self._convert_embedding_response(
            await client.embeddings.create(
                model=model,
                dimensions=kwargs.get("dimensions", NOT_GIVEN),
                **embedding_kwargs,
            )
        )

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        if not self.SUPPORTS_LIST_MODELS:
            message = f"{self.PROVIDER_NAME} does not support listing models."
            raise NotImplementedError(message)
        client = cast("OpenAI", self._get_client(sync=True))
        response = client.models.list(**kwargs)
        return self._convert_list_models_response(response)
