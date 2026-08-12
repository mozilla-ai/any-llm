from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import MissingApiKeyError

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
    from azure.core.credentials_async import AsyncTokenCredential

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


class AzureProvider(AnyLLM):
    """Azure Provider using the official Azure AI Inference SDK.

    Supports two authentication modes:

    * **API key** (default): Set ``AZURE_API_KEY`` or pass ``api_key``.
    * **Microsoft Entra ID**: Pass a ``credential`` that implements the
      ``azure.core.credentials_async.AsyncTokenCredential`` protocol
      (e.g. ``DefaultAzureCredential``). When a ``credential`` is
      provided the ``api_key`` parameter is ignored. This is the
      recommended authentication method for Microsoft Foundry deployments.
    """

    PROVIDER_NAME = "azure"
    ENV_API_KEY_NAME = "AZURE_API_KEY"
    ENV_API_BASE_NAME = "AZURE_AI_CHAT_ENDPOINT"
    PROVIDER_DOCUMENTATION_URL = (
        "https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure"
    )
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_BATCH = False
    SUPPORTS_RERANK = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    chat_client: aio.ChatCompletionsClient
    embeddings_client: aio.EmbeddingsClient

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # Allow api_key to be None when a TokenCredential is used for
        # Microsoft Entra ID authentication.
        if not api_key:
            api_key = os.getenv(self.ENV_API_KEY_NAME)
        return api_key

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        token_credential: AsyncTokenCredential | None = kwargs.pop("credential", None)

        if token_credential is not None:
            credential: AzureKeyCredential | AsyncTokenCredential = token_credential
        elif api_key:
            credential = AzureKeyCredential(api_key)
        else:
            raise MissingApiKeyError(self.PROVIDER_NAME, self.ENV_API_KEY_NAME)

        if not api_base:
            msg = (
                "For Azure, api_base is required. Check your deployment page for a URL like this - "
                "https://<model-deployment-name>.<region>.models.ai.azure.com"
            )
            raise ValueError(msg)

        self.chat_client = aio.ChatCompletionsClient(
            endpoint=api_base,
            credential=credential,
            **kwargs,
        )
        self.embeddings_client = aio.EmbeddingsClient(
            endpoint=api_base,
            credential=credential,
            **kwargs,
        )

    async def _stream_completion_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        azure_stream = cast(
            "AsyncIterable[StreamingChatCompletionsUpdate]",
            await self.chat_client.complete(
                model=model,
                messages=messages,
                **kwargs,
            ),
        )

        async for chunk in azure_stream:
            yield self._convert_completion_chunk_response(chunk)

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using Azure AI Inference SDK."""
        call_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return self._stream_completion_async(
                params.model_id,
                params.messages,
                **call_kwargs,
            )

        response: ChatCompletions = cast(
            "ChatCompletions",
            await self.chat_client.complete(
                model=params.model_id,
                messages=params.messages,
                **call_kwargs,
            ),
        )

        return self._convert_completion_response(response)

    @override
    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Create embeddings using Azure AI Inference SDK."""
        embedding_kwargs = self._convert_embedding_params({}, **kwargs)

        response: EmbeddingsResult = await self.embeddings_client.embed(
            model=model,
            input=inputs if isinstance(inputs, list) else [inputs],
            **embedding_kwargs,
        )

        return self._convert_embedding_response(response)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to Azure AI Inference format."""
        if params.reasoning_effort in ("auto", "none"):
            params.reasoning_effort = None

        azure_response_format = None
        if params.response_format:
            azure_response_format = _convert_response_format(params.response_format)

        # stream_options is an OpenAI-only knob (the Messages bridge sets it to
        # request streaming usage); the Azure AI Inference SDK does not model it
        # and forwards unknown kwargs to the transport, which rejects it.
        call_kwargs = params.model_dump(
            exclude_none=True, exclude={"model_id", "messages", "response_format", "stream_options"}
        )
        if azure_response_format:
            call_kwargs["response_format"] = azure_response_format

        call_kwargs.update(kwargs)
        return call_kwargs

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Azure ChatCompletions response to OpenAI ChatCompletion format."""
        return _convert_response(response)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Azure StreamingChatCompletionsUpdate to OpenAI ChatCompletionChunk format."""
        return _create_openai_chunk_from_azure_chunk(response)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters to Azure AI Inference format."""
        embedding_kwargs = {}
        if isinstance(params, dict):
            embedding_kwargs.update(params)
        embedding_kwargs.update(kwargs)
        return embedding_kwargs

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Azure EmbeddingsResult to OpenAI CreateEmbeddingResponse format."""
        return _create_openai_embedding_response_from_azure(response)

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Azure list models response to OpenAI format. Not supported by Azure."""
        msg = "Azure provider does not support listing models"
        raise NotImplementedError(msg)
