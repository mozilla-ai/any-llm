# Inspired by https://github.com/andrewyng/aisuite/tree/main/aisuite
import asyncio
import builtins
import importlib
import logging
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from any_llm.exceptions import MissingApiKeyError, UnsupportedProviderError
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

logger = logging.getLogger(__name__)


INSIDE_NOTEBOOK = hasattr(builtins, "__IPYTHON__")


class ProviderName(StrEnum):
    """String enum for supported providers."""

    ANTHROPIC = "anthropic"
    AWS = "aws"
    AZURE = "azure"
    AZUREOPENAI = "azureopenai"
    CEREBRAS = "cerebras"
    COHERE = "cohere"
    DATABRICKS = "databricks"
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    GOOGLE = "google"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    INCEPTION = "inception"
    LLAMA = "llama"
    LMSTUDIO = "lmstudio"
    LLAMAFILE = "llamafile"
    LLAMACPP = "llamacpp"
    MISTRAL = "mistral"
    MOONSHOT = "moonshot"
    NEBIUS = "nebius"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    PORTKEY = "portkey"
    SAMBANOVA = "sambanova"
    TOGETHER = "together"
    VOYAGE = "voyage"
    WATSONX = "watsonx"
    XAI = "xai"
    PERPLEXITY = "perplexity"

    @classmethod
    def from_string(cls, value: "str | ProviderName") -> "ProviderName":
        if isinstance(value, cls):
            return value

        formatted_value = value.strip().lower()
        try:
            return cls(formatted_value)
        except ValueError as exc:
            supported = [provider.value for provider in cls]
            raise UnsupportedProviderError(value, supported) from exc


class ApiConfig(BaseModel):
    """Configuration for the provider."""

    api_key: str | None = None
    api_base: str | None = None


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

    def __init__(self, config: ApiConfig) -> None:
        self._verify_no_missing_packages()
        self.config = self._verify_and_set_api_key(config)

    def _verify_no_missing_packages(self) -> None:
        if self.MISSING_PACKAGES_ERROR is not None:
            msg = f"{self.PROVIDER_NAME} required packages are not installed. Please install them with `pip install any-llm-sdk[{self.PROVIDER_NAME}]`"
            raise ImportError(msg) from self.MISSING_PACKAGES_ERROR

    def _verify_and_set_api_key(self, config: ApiConfig) -> ApiConfig:
        # Standardized API key handling. Splitting into its own function so that providers
        # Can easily override this method if they don't want verification (for instance, LMStudio)
        if not config.api_key:
            config.api_key = os.getenv(self.ENV_API_KEY_NAME)

        if not config.api_key:
            raise MissingApiKeyError(self.PROVIDER_NAME, self.ENV_API_KEY_NAME)
        return config

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


class ProviderFactory:
    """Factory to dynamically load provider instances based on the naming conventions."""

    PROVIDERS_DIR = Path(__file__).parent / "providers"

    @classmethod
    def create_provider(cls, provider_key: str | ProviderName, config: ApiConfig) -> Provider:
        """Dynamically load and create an instance of a provider based on the naming convention."""
        provider_key = ProviderName.from_string(provider_key).value

        provider_class_name = f"{provider_key.capitalize()}Provider"
        provider_module_name = f"{provider_key}"

        module_path = f"any_llm.providers.{provider_module_name}"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            msg = f"Could not import module {module_path}: {e!s}. Please ensure the provider is supported by doing ProviderFactory.get_supported_providers()"
            raise ImportError(msg) from e

        provider_class: type[Provider] = getattr(module, provider_class_name)
        return provider_class(config=config)

    @classmethod
    def get_provider_class(cls, provider_key: str | ProviderName) -> type[Provider]:
        """Get the provider class without instantiating it.

        Args:
            provider_key: The provider key (e.g., 'anthropic', 'openai')

        Returns:
            The provider class
        """
        provider_key = ProviderName.from_string(provider_key).value

        provider_class_name = f"{provider_key.capitalize()}Provider"
        provider_module_name = f"{provider_key}"

        module_path = f"any_llm.providers.{provider_module_name}"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            msg = f"Could not import module {module_path}: {e!s}. Please ensure the provider is supported by doing ProviderFactory.get_supported_providers()"
            raise ImportError(msg) from e

        provider_class: type[Provider] = getattr(module, provider_class_name)
        return provider_class

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get a list of supported provider keys."""
        return [provider.value for provider in ProviderName]

    @classmethod
    def get_all_provider_metadata(cls) -> list[ProviderMetadata]:
        """Get metadata for all supported providers.

        Returns:
            List of dictionaries containing provider metadata
        """
        providers: list[ProviderMetadata] = []
        for provider_key in cls.get_supported_providers():
            provider_class = cls.get_provider_class(provider_key)
            metadata = provider_class.get_provider_metadata()
            providers.append(metadata)

        # Sort providers by name
        providers.sort(key=lambda x: x.name)
        return providers

    @classmethod
    def get_provider_enum(cls, provider_key: str) -> ProviderName:
        """Convert a string provider key to a ProviderName enum."""
        try:
            return ProviderName(provider_key)
        except ValueError as e:
            supported = [provider.value for provider in ProviderName]
            raise UnsupportedProviderError(provider_key, supported) from e

    @classmethod
    def split_model_provider(cls, model: str) -> tuple[ProviderName, str]:
        """Extract the provider key from the model identifier.

        Supports both new format 'provider:model' (e.g., 'mistral:mistral-small')
        and legacy format 'provider/model' (e.g., 'mistral/mistral-small').

        The legacy format will be deprecated in version 1.0.
        """
        colon_index = model.find(":")
        slash_index = model.find("/")

        # Determine which delimiter comes first
        if colon_index != -1 and (slash_index == -1 or colon_index < slash_index):
            # The colon came first, so it's using the new syntax.
            provider, model_name = model.split(":", 1)
        elif slash_index != -1:
            # Slash comes first, so it's the legacy syntax
            warnings.warn(
                f"Model format 'provider/model' is deprecated and will be removed in version 1.0. "
                f"Please use 'provider:model' format instead. Got: '{model}'",
                DeprecationWarning,
                stacklevel=3,
            )
            provider, model_name = model.split("/", 1)
        else:
            msg = f"Invalid model format. Expected 'provider:model' or 'provider/model', got '{model}'"
            raise ValueError(msg)

        if not provider or not model_name:
            msg = f"Invalid model format. Expected 'provider:model' or 'provider/model', got '{model}'"
            raise ValueError(msg)
        return cls.get_provider_enum(provider), model_name
