from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

try:
    from mistralai import Mistral
    from mistralai.extra import response_format_from_pydantic_model

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from pydantic import BaseModel

from any_llm.provider import Provider
from any_llm.providers.mistral.utils import (
    _create_mistral_completion_from_response,
    _create_openai_chunk_from_mistral_chunk,
    _create_openai_embedding_response_from_mistral,
    _patch_messages,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse

if TYPE_CHECKING:
    from mistralai.models.embeddingresponse import EmbeddingResponse


class MistralProvider(Provider):
    """Mistral Provider using the new response conversion utilities."""

    PROVIDER_NAME = "mistral"
    ENV_API_KEY_NAME = "MISTRAL_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.mistral.ai/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = True

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    def _stream_completion(
        self,
        client: Mistral,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        # Get the Mistral stream
        mistral_stream = client.chat.stream(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        for event in mistral_stream:
            yield _create_openai_chunk_from_mistral_chunk(event)

    async def _stream_completion_async(
        self, client: Mistral, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        mistral_stream = await client.chat.stream_async(model=model, messages=messages, **kwargs)  # type: ignore[arg-type]

        async for event in mistral_stream:
            yield _create_openai_chunk_from_mistral_chunk(event)

    async def acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        patched_messages = _patch_messages(params.messages)

        if (
            params.response_format is not None
            and isinstance(params.response_format, type)
            and issubclass(params.response_format, BaseModel)
        ):
            kwargs["response_format"] = response_format_from_pydantic_model(params.response_format)

        client = Mistral(api_key=self.config.api_key, server_url=self.config.api_base)

        if params.stream:
            return self._stream_completion_async(
                client,
                params.model_id,
                patched_messages,
                **params.model_dump(
                    exclude_none=True, exclude={"model_id", "messages", "reasoning_effort", "response_format", "stream"}
                ),
                **kwargs,
            )

        response = await client.chat.complete_async(
            model=params.model_id,
            messages=patched_messages,  # type: ignore[arg-type]
            **params.model_dump(
                exclude_none=True, exclude={"model_id", "messages", "reasoning_effort", "response_format", "stream"}
            ),
            **kwargs,
        )

        return _create_mistral_completion_from_response(
            response_data=response,
            model=params.model_id,
        )

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Mistral."""
        client = Mistral(api_key=self.config.api_key, server_url=self.config.api_base)

        patched_messages = _patch_messages(params.messages)

        if (
            params.response_format is not None
            and isinstance(params.response_format, type)
            and issubclass(params.response_format, BaseModel)
        ):
            kwargs["response_format"] = response_format_from_pydantic_model(params.response_format)

        if not params.stream:
            response = client.chat.complete(
                model=params.model_id,
                messages=patched_messages,  # type: ignore[arg-type]
                **params.model_dump(
                    exclude_none=True, exclude={"model_id", "messages", "reasoning_effort", "response_format", "stream"}
                ),
                **kwargs,
            )

            return _create_mistral_completion_from_response(
                response_data=response,
                model=params.model_id,
            )

        return self._stream_completion(
            client,
            params.model_id,
            patched_messages,
            **params.model_dump(
                exclude_none=True, exclude={"model_id", "messages", "reasoning_effort", "response_format", "stream"}
            ),
            **kwargs,
        )

    def embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        client = Mistral(api_key=self.config.api_key, server_url=self.config.api_base)

        result: EmbeddingResponse = client.embeddings.create(
            model=model,
            inputs=inputs,
            **kwargs,
        )

        return _create_openai_embedding_response_from_mistral(result)

    async def aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        client = Mistral(api_key=self.config.api_key, server_url=self.config.api_base)
        result: EmbeddingResponse = await client.embeddings.create_async(
            model=model,
            inputs=inputs,
            **kwargs,
        )

        return _create_openai_embedding_response_from_mistral(result)
