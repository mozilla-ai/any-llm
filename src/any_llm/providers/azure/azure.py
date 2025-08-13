import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

try:
    from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
    from azure.core.credentials import AzureKeyCredential

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from any_llm.provider import ApiConfig, Provider
from any_llm.providers.azure.utils import (
    _convert_response,
    _convert_response_format,
    _create_openai_chunk_from_azure_chunk,
    _create_openai_embedding_response_from_azure,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CreateEmbeddingResponse

if TYPE_CHECKING:
    from azure.ai.inference.models import ChatCompletions, EmbeddingsResult, StreamingChatCompletionsUpdate


class AzureProvider(Provider):
    """Azure Provider using the official Azure AI Inference SDK."""

    PROVIDER_NAME = "azure"
    ENV_API_KEY_NAME = "AZURE_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://azure.microsoft.com/en-us/products/ai-services/openai-service"
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False

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

    def _create_chat_client(self) -> ChatCompletionsClient:
        """Create and configure a ChatCompletionsClient."""
        return ChatCompletionsClient(
            endpoint=self._get_endpoint(),
            credential=AzureKeyCredential(self.config.api_key or ""),
            api_version=self.api_version,
        )

    def _create_embeddings_client(self) -> EmbeddingsClient:
        """Create and configure an EmbeddingsClient."""
        return EmbeddingsClient(
            endpoint=self._get_endpoint(),
            credential=AzureKeyCredential(self.config.api_key or ""),
            api_version=self.api_version,
        )

    def _stream_completion(
        self,
        client: ChatCompletionsClient,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        azure_stream: Iterator[StreamingChatCompletionsUpdate] = client.complete(
            model=model,
            messages=messages,
            **kwargs,
        )

        for chunk in azure_stream:
            yield _create_openai_chunk_from_azure_chunk(chunk)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Azure AI Inference SDK."""
        client: ChatCompletionsClient = self._create_chat_client()

        if "response_format" in kwargs:
            kwargs["response_format"] = _convert_response_format(kwargs["response_format"])

        if kwargs.get("stream", False):
            return self._stream_completion(client, model, messages, **kwargs)
        response: ChatCompletions = client.complete(
            model=model,
            messages=messages,
            **kwargs,
        )

        return _convert_response(response)

    def embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Create embeddings using Azure AI Inference SDK."""
        client: EmbeddingsClient = self._create_embeddings_client()

        input_list: list[str]
        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs

        response: EmbeddingsResult = client.embed(
            model=model,
            input=input_list,
            **kwargs,
        )

        return _create_openai_embedding_response_from_azure(response)
