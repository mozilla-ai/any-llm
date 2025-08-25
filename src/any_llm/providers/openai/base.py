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
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True

    PACKAGES_INSTALLED = True

    _DEFAULT_REASONING_EFFORT: Literal["minimal", "low", "medium", "high", "auto"] | None = None

    @property
    def openai_client(self) -> OpenAI:
        return OpenAI(
            base_url=self.config.api_base or self.API_BASE or os.getenv("OPENAI_API_BASE"),
            api_key=self.config.api_key,
        )

    @property
    def async_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.config.api_base or self.API_BASE or os.getenv("OPENAI_API_BASE"),
            api_key=self.config.api_key,
        )

    def _convert_completion_response_async(
        self, response: OpenAIChatCompletion | AsyncStream[OpenAIChatCompletionChunk]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Convert an OpenAI completion response to an AnyLLM completion response."""
        if isinstance(response, OpenAIChatCompletion):
            return _convert_chat_completion(response)

        async def chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in response:
                if not isinstance(chunk.created, int):
                    logger.warning(
                        "API returned an unexpected created type: %s. Setting to int.",
                        type(chunk.created),
                    )
                    chunk.created = int(chunk.created)

                normalized_chunk = _normalize_openai_dict_response(chunk.model_dump())
                yield ChatCompletionChunk.model_validate(normalized_chunk)

        return chunk_iterator()

    async def acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        client = self.async_openai_client

        if params.reasoning_effort == "auto":
            params.reasoning_effort = self._DEFAULT_REASONING_EFFORT

        if params.response_format:
            if params.stream:
                msg = "stream is not supported for response_format"
                raise ValueError(msg)

            response = await client.chat.completions.parse(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "stream"}),
                **kwargs,
            )
        else:
            response = await client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages"}),
                **kwargs,
            )
        return self._convert_completion_response_async(response)

    async def aresponses(
        self, model: str, input_data: Any, **kwargs: Any
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """Call OpenAI Responses API"""
        client = self.async_openai_client

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

        client = self.async_openai_client

        return await client.embeddings.create(
            model=model,
            input=inputs,
            dimensions=kwargs.get("dimensions", NOT_GIVEN),
            **kwargs,
        )

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        if not self.SUPPORTS_LIST_MODELS:
            message = f"{self.PROVIDER_NAME} does not support listing models."
            raise NotImplementedError(message)
        client = self.openai_client

        return client.models.list(**kwargs).data
