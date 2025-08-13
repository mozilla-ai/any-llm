# Inspired by https://github.com/andrewyng/aisuite/tree/main/aisuite
import asyncio
import importlib
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from any_llm.exceptions import MissingApiKeyError, UnsupportedProviderError
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CreateEmbeddingResponse,
)
from any_llm.types.provider import ProviderMetadata
from any_llm.types.responses import Response, ResponseInputParam, ResponseStreamEvent


class ProviderName(str, Enum):
    """String enum for supported providers."""

    ANTHROPIC = "anthropic"
    AWS = "aws"
    AZURE = "azure"
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
    MISTRAL = "mistral"
    MOONSHOT = "moonshot"
    NEBIUS = "nebius"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    PORTKEY = "portkey"
    SAMBANOVA = "sambanova"
    TOGETHER = "together"
    WATSONX = "watsonx"
    XAI = "xai"


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

    API_BASE: str | None = None
    """This is used to set the API base for the provider.
    It is not required but may prove useful for providers that have overridable api bases.
    """

    # === Internal Flag Checks ===
    PACKAGES_INSTALLED: bool
    """Some providers use SDKs that are not installed by default.
    This flag is used to check if the packages are installed before instantiating the provider.
    """

    def __init__(self, config: ApiConfig) -> None:
        if not self.PACKAGES_INSTALLED:
            msg = f"{self.PROVIDER_NAME} required packages are not installed. Please install them with `pip install any-llm-sdk[{self.PROVIDER_NAME}]`"
            raise ImportError(msg)
        self.config = self._verify_and_set_api_key(config)

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
            class_name=cls.__name__,
        )

    @abstractmethod
    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """This method is designed to make the API call to the provider.

        Args:
            model: The model to use
            messages: The messages to send
            kwargs: The kwargs to pass to the API call

        Returns:
            The response from the API call
        """
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        return await asyncio.to_thread(self.completion, model, messages, **kwargs)

    def responses(
        self, model: str, input_data: str | ResponseInputParam, **kwargs: Any
    ) -> Response | Iterator[ResponseStreamEvent]:
        """Create a response using the provider's Responses API if supported.

        Default implementation raises NotImplementedError. Providers that set
        SUPPORTS_RESPONSES to True must override this method.
        """
        msg = "This provider does not support the Responses API."
        raise NotImplementedError(msg)

    async def aresponses(
        self, model: str, input_data: str | ResponseInputParam, **kwargs: Any
    ) -> Response | Iterator[ResponseStreamEvent]:
        return await asyncio.to_thread(self.responses, model, input_data, **kwargs)

    def embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    async def aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        return await asyncio.to_thread(self.embedding, model, inputs, **kwargs)


class ProviderFactory:
    """Factory to dynamically load provider instances based on the naming conventions."""

    PROVIDERS_DIR = Path(__file__).parent / "providers"

    @classmethod
    def create_provider(cls, provider_key: str | ProviderName, config: ApiConfig) -> Provider:
        """Dynamically load and create an instance of a provider based on the naming convention."""
        if isinstance(provider_key, ProviderName):
            provider_key = provider_key.value

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
        if isinstance(provider_key, ProviderName):
            provider_key = provider_key.value

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
        """Extract the provider key from the model identifier, e.g., "mistral/mistral-small"""
        if "/" not in model:
            msg = f"Invalid model format. Expected 'provider/model', got '{model}'"
            raise ValueError(msg)
        provider, model = model.split("/", 1)

        if not provider or not model:
            msg = f"Invalid model format. Expected 'provider/model', got '{model}'"
            raise ValueError(msg)
        return cls.get_provider_enum(provider), model
