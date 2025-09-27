# Inspired by https://github.com/andrewyng/aisuite/tree/main/aisuite
import asyncio
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from any_llm.config import ClientConfig
from any_llm.constants import INSIDE_NOTEBOOK
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
    CreateEmbeddingResponse,
)
from any_llm.types.model import Model
from any_llm.types.provider import ProviderMetadata
from any_llm.types.responses import Response, ResponseInputParam, ResponseStreamEvent
from any_llm.utils.aio import async_iter_to_sync_iter, run_async_in_sync


class Provider(ABC):
    """Provider for the LLM."""

    # === Provider-specific configuration (to be overridden by subclasses) ===
    PROVIDER_NAME: str
    """Must match the name of the provider directory  (case sensitive)"""

    PROVIDER_DOCUMENTATION_URL: str
    """Link to the provider's documentation"""

    ENV_API_KEY_NAME: str
    """Environment variable name for the API key"""

    # === Feature support flags (to be set by subclasses) ===
    SUPPORTS_COMPLETION_STREAMING: bool
    """OpenAI Streaming Completion API"""

    SUPPORTS_COMPLETION: bool
    """OpenAI Completion API"""

    SUPPORTS_COMPLETION_REASONING: bool
    """Reasoning Content attached to Completion API Response"""

    SUPPORTS_COMPLETION_IMAGE: bool
    """Image Support for Completion API"""

    SUPPORTS_COMPLETION_PDF: bool
    """PDF Support for Completion API"""

    SUPPORTS_EMBEDDING: bool
    """OpenAI Embedding API"""

    SUPPORTS_RESPONSES: bool
    """OpenAI Responses API"""

    SUPPORTS_LIST_MODELS: bool
    """OpenAI Models API"""

    API_BASE: str | None = None
    """This is used to set the API base for the provider.
    It is not required but may prove useful for providers that have overridable api bases.
    """

    # === Internal Flag Checks ===
    MISSING_PACKAGES_ERROR: ImportError | None = None
    """Some providers use SDKs that are not installed by default.
    This flag is used to check if the packages are installed before instantiating the provider.
    """

    def __init__(self, config: ClientConfig) -> None:
        self._verify_no_missing_packages()
        self.config = self._verify_and_set_api_key(config)

    def _verify_no_missing_packages(self) -> None:
        if self.MISSING_PACKAGES_ERROR is not None:
            msg = f"{self.PROVIDER_NAME} required packages are not installed. Please install them with `pip install any-llm-sdk[{self.PROVIDER_NAME}]`"
            raise ImportError(msg) from self.MISSING_PACKAGES_ERROR

    def _verify_and_set_api_key(self, config: ClientConfig) -> ClientConfig:
        # Standardized API key handling. Splitting into its own function so that providers
        # Can easily override this method if they don't want verification (for instance, LMStudio)
        if not config.api_key:
            config.api_key = os.getenv(self.ENV_API_KEY_NAME)

        if not config.api_key:
            raise MissingApiKeyError(self.PROVIDER_NAME, self.ENV_API_KEY_NAME)
        return config

    @staticmethod
    @abstractmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @staticmethod
    @abstractmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @staticmethod
    @abstractmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @staticmethod
    @abstractmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @staticmethod
    @abstractmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @staticmethod
    @abstractmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @classmethod
    def get_provider_metadata(cls) -> ProviderMetadata:
        """Get provider metadata without requiring instantiation.

        Returns:
            Dictionary containing provider metadata including name, environment variable,
            documentation URL, and class name.
        """
        return ProviderMetadata(
            name=cls.PROVIDER_NAME,
            env_key=cls.ENV_API_KEY_NAME,
            doc_url=cls.PROVIDER_DOCUMENTATION_URL,
            streaming=cls.SUPPORTS_COMPLETION_STREAMING,
            reasoning=cls.SUPPORTS_COMPLETION_REASONING,
            completion=cls.SUPPORTS_COMPLETION,
            image=cls.SUPPORTS_COMPLETION_IMAGE,
            pdf=cls.SUPPORTS_COMPLETION_PDF,
            embedding=cls.SUPPORTS_EMBEDDING,
            responses=cls.SUPPORTS_RESPONSES,
            list_models=cls.SUPPORTS_LIST_MODELS,
            class_name=cls.__name__,
        )

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """This method is designed to make the API call to the provider.

        Args:
            params: The completion parameters
            kwargs: Extra kwargs to pass to the API call

        Returns:
            The response from the API call
        """
        response = run_async_in_sync(self.acompletion(params, **kwargs), allow_running_loop=INSIDE_NOTEBOOK)
        if isinstance(response, ChatCompletion):
            return response

        return async_iter_to_sync_iter(response)

    @abstractmethod
    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def responses(
        self, model: str, input_data: str | ResponseInputParam, **kwargs: Any
    ) -> Response | Iterator[ResponseStreamEvent]:
        """Create a response using the provider's Responses API if supported.

        Default implementation raises NotImplementedError. Providers that set
        SUPPORTS_RESPONSES to True must override this method.
        """
        response = run_async_in_sync(self.aresponses(model, input_data, **kwargs), allow_running_loop=INSIDE_NOTEBOOK)
        if isinstance(response, Response):
            return response
        return async_iter_to_sync_iter(response)

    async def aresponses(
        self, model: str, input_data: str | ResponseInputParam, **kwargs: Any
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        return run_async_in_sync(self.aembedding(model, inputs, **kwargs), allow_running_loop=INSIDE_NOTEBOOK)

    async def aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Return a list of Model if the provider supports listing models.
        Should be overridden by subclasses.
        """
        msg = "Subclasses must implement list_models method"
        if not self.SUPPORTS_LIST_MODELS:
            raise NotImplementedError(msg)
        raise NotImplementedError(msg)

    async def list_models_async(self, **kwargs: Any) -> Sequence[Model]:
        return await asyncio.to_thread(self.list_models, **kwargs)
