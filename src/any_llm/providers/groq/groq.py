from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai.types.responses import Response as OpenAIResponse
from pydantic import BaseModel

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import UnsupportedParameterError

if TYPE_CHECKING:
    from openai.types.responses import ResponseStreamEvent as OpenAIResponseStreamEvent
    from openresponses_types import CreateResponseBody, ResponseResource

    from any_llm.types.completion import CreateEmbeddingResponse

MISSING_PACKAGES_ERROR = None
try:
    from groq import AsyncGroq

    from .utils import (
        _convert_models_list,
        _create_openai_chunk_from_groq_chunk,
        to_chat_completion,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from groq import AsyncStream as GroqAsyncStream
    from groq.types.chat import ChatCompletion as GroqChatCompletion
    from groq.types.chat import ChatCompletionChunk as GroqChatCompletionChunk

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
    from any_llm.types.model import Model


class GroqProvider(AnyLLM):
    """Groq Provider."""

    PROVIDER_NAME = "groq"
    ENV_API_KEY_NAME = "GROQ_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://groq.com/api"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False

    API_BASE = "https://api.groq.com/openai/v1"

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncGroq
    openai_client: AsyncOpenAI

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Groq API."""
        # Groq does not support providing reasoning effort
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages"})
        if converted_params.get("reasoning_effort") in ("auto", "none"):
            converted_params.pop("reasoning_effort")
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Groq response to OpenAI format."""
        return to_chat_completion(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Groq chunk response to OpenAI format."""
        return _create_openai_chunk_from_groq_chunk(response)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Groq."""
        msg = "Groq does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Groq embedding response to OpenAI format."""
        msg = "Groq does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Groq list models response to OpenAI format."""
        return _convert_models_list(response)

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self.kwargs = kwargs
        self.client = AsyncGroq(api_key=api_key, **kwargs)
        self.openai_client = AsyncOpenAI(
            base_url=api_base or self.API_BASE,
            api_key=api_key,
            **kwargs,
        )

    async def _stream_async_completion(
        self, params: CompletionParams, **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        if params.stream and params.response_format:
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)

        completion_kwargs = self._convert_completion_params(params, **kwargs)
        stream: GroqAsyncStream[GroqChatCompletionChunk] = await self.client.chat.completions.create(
            model=params.model_id,
            messages=cast("Any", params.messages),
            **completion_kwargs,
        )

        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in stream:
                yield self._convert_completion_chunk_response(chunk)

        return _stream()

    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if params.response_format:
            if isinstance(params.response_format, type) and issubclass(params.response_format, BaseModel):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": params.response_format.__name__,
                        "schema": params.response_format.model_json_schema(),
                    },
                }
            else:
                kwargs["response_format"] = params.response_format

        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return await self._stream_async_completion(
                params,
                **kwargs,
            )
        response: GroqChatCompletion = await self.client.chat.completions.create(
            model=params.model_id,
            messages=cast("Any", params.messages),
            **completion_kwargs,
        )

        return self._convert_completion_response(response)

    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models_list = await self.client.models.list(**kwargs)
        return self._convert_list_models_response(models_list)

    async def _aresponses(
        self, params: CreateResponseBody, **kwargs: Any
    ) -> ResponseResource | OpenAIResponse | AsyncIterator[dict[str, Any]]:
        """Call Groq Responses API and return OpenAI Response type directly.

        Groq is not yet compliant with the OpenResponses spec, so we return
        the OpenAI Response type directly instead of converting to ResponseResource.
        """
        response: OpenAIResponse | AsyncStream[OpenAIResponseStreamEvent] = await self.openai_client.responses.create(
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
