import os
from typing import Any, Iterator, Union

try:
    from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
    from azure.ai.inference.models import ChatCompletions, EmbeddingsResult, StreamingChatCompletionsUpdate
    from azure.core.credentials import AzureKeyCredential
except ImportError as exc:
    msg = "azure-ai-inference is not installed. Please install it with `pip install any-llm-sdk[azure]`"
    raise ImportError(msg) from exc

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CreateEmbeddingResponse

from any_llm.provider import Provider, ApiConfig
from any_llm.providers.azure.utils import (
    _convert_response,
    _create_openai_chunk_from_azure_chunk,
    _create_openai_embedding_response_from_azure,
    _convert_response_format,
)


class AzureProvider(Provider):
    """Azure Provider using the official Azure AI Inference SDK."""

    PROVIDER_NAME: str = "Azure"
    ENV_API_KEY_NAME: str = "AZURE_API_KEY"
    PROVIDER_DOCUMENTATION_URL: str = "https://azure.microsoft.com/en-us/products/ai-services/openai-service"

    SUPPORTS_STREAMING: bool = True
    SUPPORTS_EMBEDDING: bool = True
    SUPPORTS_REASONING: bool = False
    SUPPORTS_COMPLETION: bool = True

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Azure provider."""
        super().__init__(config)
        self.api_version: str = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")

    def _get_endpoint(self) -> str:
        """Get the Azure endpoint URL."""
        if self.config.api_base:
            return self.config.api_base

        raise ValueError(
            "For Azure, api_base is required. Check your deployment page for a URL like this - "
            "https://<model-deployment-name>.<region>.models.ai.azure.com"
        )

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

    @classmethod
    def verify_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Azure provider."""
        # No specific validation needed for Azure provider currently

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Create a chat completion using Azure AI Inference SDK."""
        client: ChatCompletionsClient = self._create_chat_client()

        # Handle response_format conversion for Pydantic models
        if "response_format" in kwargs:
            kwargs["response_format"] = _convert_response_format(kwargs["response_format"])

        # Handle streaming vs non-streaming
        if kwargs.get("stream", False):
            return self._stream_completion(client, model, messages, **kwargs)
        else:
            response: ChatCompletions = client.complete(
                model=model,
                messages=messages,
                **kwargs,
            )

            return _convert_response(response)

    def embedding(
        self,
        model: str,
        inputs: Union[str, list[str]],
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
