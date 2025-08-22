import os
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, cast

try:
    from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient, aio
    from azure.ai.projects import AIProjectClient
    from azure.core.credentials import AzureKeyCredential
    from azure.identity import DefaultAzureCredential

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from any_llm.exceptions import UnsupportedModelResponseError
from any_llm.provider import ApiConfig, Provider
from any_llm.providers.azure.utils import (
    _convert_models_list,
    _convert_response,
    _convert_response_format,
    _create_openai_chunk_from_azure_chunk,
    _create_openai_embedding_response_from_azure,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.types.model import Model

if TYPE_CHECKING:
    from azure.ai.inference.models import ChatCompletions, EmbeddingsResult, StreamingChatCompletionsUpdate


class AzureProvider(Provider):
    """Azure Provider using the official Azure AI Inference SDK."""

    PROVIDER_LABEL = "Azure"
    PROVIDER_NAME = "azure"
    ENV_API_KEY_NAME = "AZURE_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://azure.microsoft.com/en-us/products/ai-services/openai-service"
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_LIST_MODELS = True

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Azure provider."""
        super().__init__(config)
        self.api_version: str = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")

    def _get_endpoint(self) -> str:
        """Get the Azure endpoint URL."""
        if self.config.api_base:
            return self.config.api_base

        msg = (
            "For Azure, api_base is required. Check your deployment page for a URL like this - "
            "https://<model-deployment-name>.<region>.models.ai.azure.com"
        )
        raise ValueError(msg)

    def _create_chat_client_async(self) -> aio.ChatCompletionsClient:
        """Create and configure a ChatCompletionsClient."""
        return aio.ChatCompletionsClient(
            endpoint=self._get_endpoint(),
            credential=AzureKeyCredential(self.config.api_key or ""),
            api_version=self.api_version,
        )

    def _create_embeddings_client_async(self) -> aio.EmbeddingsClient:
        """Create and configure an EmbeddingsClient."""
        return aio.EmbeddingsClient(
            endpoint=self._get_endpoint(),
            credential=AzureKeyCredential(self.config.api_key or ""),
            api_version=self.api_version,
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
            yield _create_openai_chunk_from_azure_chunk(chunk)

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using Azure AI Inference SDK."""
        client: aio.ChatCompletionsClient = self._create_chat_client_async()

        if params.reasoning_effort == "auto":
            params.reasoning_effort = None

        azure_response_format = None
        if params.response_format:
            azure_response_format = _convert_response_format(params.response_format)

        call_kwargs = params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format"})
        if azure_response_format:
            call_kwargs["response_format"] = azure_response_format

        if params.stream:
            return self._stream_completion_async(
                client,
                params.model_id,
                params.messages,
                **call_kwargs,
                **kwargs,
            )

        response: ChatCompletions = cast(
            "ChatCompletions",
            await client.complete(
                model=params.model_id,
                messages=params.messages,
                **call_kwargs,
                **kwargs,
            ),
        )

        return _convert_response(response)

    async def aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Create embeddings using Azure AI Inference SDK."""
        client: aio.EmbeddingsClient = self._create_embeddings_client_async()

        response: EmbeddingsResult = await client.embed(
            model=model,
            input=inputs if isinstance(inputs, list) else [inputs],
            **kwargs,
        )

        return _create_openai_embedding_response_from_azure(response)

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Return a list of Model for Azure OpenAI models.
        Handles all errors and raises ProviderError on failure.
        """
        try:
            project_client = AIProjectClient(
                endpoint=self._get_endpoint(),
                credential=DefaultAzureCredential(),
            )
            # documentation - https://learn.microsoft.com/en-us/azure/ai-foundry/quickstarts/get-started-code?source=recommendations&tabs=python&pivots=fdp-project
            azure_openai_client = project_client.get_openai_client(api_version=self.api_version)
            azure_models = azure_openai_client.models.list(**kwargs)
            return _convert_models_list(azure_models, self.PROVIDER_NAME)
        except Exception as e:
            raise UnsupportedModelResponseError(
                message="Failed to parse Azure model response.", original_exception=e
            ) from e
