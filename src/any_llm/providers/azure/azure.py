from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

from any_llm.provider import Provider

MISSING_PACKAGES_ERROR = None
try:
    from azure.ai.inference import aio
    from azure.core.credentials import AzureKeyCredential

    from .utils import (
        _convert_response,
        _convert_response_format,
        _create_openai_chunk_from_azure_chunk,
        _create_openai_embedding_response_from_azure,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Sequence

    from azure.ai.inference import aio  # noqa: TC004
    from azure.ai.inference.models import ChatCompletions, EmbeddingsResult, StreamingChatCompletionsUpdate

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


class AzureProvider(Provider):
    """Azure Provider using the official Azure AI Inference SDK."""

    PROVIDER_NAME = "azure"
    ENV_API_KEY_NAME = "AZURE_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://azure.microsoft.com/en-us/products/ai-services/openai-service"
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_LIST_MODELS = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    def _get_endpoint(self) -> str:
        """Get the Azure endpoint URL."""
        if self.config.api_base:
            return self.config.api_base

        msg = (
            "For Azure, api_base is required. Check your deployment page for a URL like this - "
            "https://<model-deployment-name>.<region>.models.ai.azure.com"
        )
        raise ValueError(msg)

    def _create_chat_client_async(self, api_version: str) -> aio.ChatCompletionsClient:
        """Create and configure a ChatCompletionsClient."""
        return aio.ChatCompletionsClient(
            endpoint=self._get_endpoint(),
            credential=AzureKeyCredential(self.config.api_key or ""),
            api_version=api_version,
            **(self.config.client_args if self.config.client_args else {}),
        )

    def _create_embeddings_client_async(self, api_version: str) -> aio.EmbeddingsClient:
        """Create and configure an EmbeddingsClient."""
        return aio.EmbeddingsClient(
            endpoint=self._get_endpoint(),
            credential=AzureKeyCredential(self.config.api_key or ""),
            api_version=api_version,
            **(self.config.client_args if self.config.client_args else {}),
        )

    async def _stream_completion_async(
        self,
        client: aio.ChatCompletionsClient,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        azure_stream = cast(
            "AsyncIterable[StreamingChatCompletionsUpdate]",
            await client.complete(
                model=model,
                messages=messages,
                **kwargs,
            ),
        )

        async for chunk in azure_stream:
            yield self._convert_completion_chunk_response(chunk)

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using Azure AI Inference SDK."""
        api_version = os.getenv("AZURE_API_VERSION", kwargs.pop("api_version", "2024-02-15-preview"))
        client: aio.ChatCompletionsClient = self._create_chat_client_async(api_version)

        call_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return self._stream_completion_async(
                client,
                params.model_id,
                params.messages,
                **call_kwargs,
            )

        response: ChatCompletions = cast(
            "ChatCompletions",
            await client.complete(
                model=params.model_id,
                messages=params.messages,
                **call_kwargs,
            ),
        )

        return self._convert_completion_response(response)

    async def aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Create embeddings using Azure AI Inference SDK."""
        api_version = os.getenv("AZURE_API_VERSION", kwargs.pop("api_version", "2024-02-15-preview"))
        client: aio.EmbeddingsClient = self._create_embeddings_client_async(api_version)

        embedding_kwargs = self._convert_embedding_params({}, **kwargs)

        response: EmbeddingsResult = await client.embed(
            model=model,
            input=inputs if isinstance(inputs, list) else [inputs],
            **embedding_kwargs,
        )

        return self._convert_embedding_response(response)

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to Azure AI Inference format."""
        if params.reasoning_effort == "auto":
            params.reasoning_effort = None

        azure_response_format = None
        if params.response_format:
            azure_response_format = _convert_response_format(params.response_format)

        call_kwargs = params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format"})
        if azure_response_format:
            call_kwargs["response_format"] = azure_response_format

        call_kwargs.update(kwargs)
        return call_kwargs

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Azure ChatCompletions response to OpenAI ChatCompletion format."""
        return _convert_response(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Azure StreamingChatCompletionsUpdate to OpenAI ChatCompletionChunk format."""
        return _create_openai_chunk_from_azure_chunk(response)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters to Azure AI Inference format."""
        embedding_kwargs = {}
        if isinstance(params, dict):
            embedding_kwargs.update(params)
        embedding_kwargs.update(kwargs)
        return embedding_kwargs

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Azure EmbeddingsResult to OpenAI CreateEmbeddingResponse format."""
        return _create_openai_embedding_response_from_azure(response)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Azure list models response to OpenAI format. Not supported by Azure."""
        msg = "Azure provider does not support listing models"
        raise NotImplementedError(msg)
