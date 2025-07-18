# Inspired by https://github.com/andrewyng/aisuite/tree/main/aisuite
import asyncio
import importlib
import json
from abc import ABC, abstractmethod
from enum import Enum
import os
from pathlib import Path
from typing import Any, Type, Union

from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from pydantic import BaseModel

from any_llm.exceptions import MissingApiKeyError, UnsupportedProviderError


def convert_instructor_response(instructor_response: Any, model: str, provider_name: str) -> ChatCompletion:
    """
    Convert instructor response to ChatCompletion format.

    Args:
        instructor_response: The response from instructor
        model: The model name used
        provider_name: The provider name (used in the response ID)

    Returns:
        ChatCompletion object with the structured response as JSON content
    """
    # Convert the structured response to JSON string
    if hasattr(instructor_response, "model_dump"):
        content = json.dumps(instructor_response.model_dump())
    else:
        content = json.dumps(instructor_response)

    # Create a mock ChatCompletion response
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
    )

    choice = Choice(
        finish_reason="stop",
        index=0,
        message=message,
    )

    return ChatCompletion(
        id=f"{provider_name}-instructor-response",
        choices=[choice],
        created=0,
        model=model,
        object="chat.completion",
    )


class ProviderName(str, Enum):
    """String enum for supported providers."""

    ANTHROPIC = "anthropic"
    AWS = "aws"
    AZURE = "azure"
    CEREBRAS = "cerebras"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    GOOGLE = "google"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    INCEPTION = "inception"
    MISTRAL = "mistral"
    MOONSHOT = "moonshot"
    NEBIUS = "nebius"
    OLLAMA = "ollama"
    OPENAI = "openai"
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

    # Provider-specific configuration (to be overridden by subclasses)
    PROVIDER_NAME: str
    ENV_API_KEY_NAME: str

    def __init__(self, config: ApiConfig) -> None:
        self.config = config
        # Standardized API key handling
        if not config.api_key:
            config.api_key = os.getenv(self.ENV_API_KEY_NAME)

        if not config.api_key:
            raise MissingApiKeyError(self.PROVIDER_NAME, self.ENV_API_KEY_NAME)

    @abstractmethod
    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """This method is designed to check whether a provider supports specific arguments.
        It is not used to verify the API key.

        Args:
            kwargs: The kwargs to check

        Returns:
            None

        Raises:
            UnsupportedParameterError: If the provider does not support the argument
        """
        return None

    @abstractmethod
    def _make_api_call(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """This method is designed to make the API call to the provider.

        Args:
            model: The model to use
            messages: The messages to send
            kwargs: The kwargs to pass to the API call

        Returns:
            The response from the API call
        """
        raise NotImplementedError("Subclasses must implement this method")

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        self.verify_kwargs(kwargs)
        return self._make_api_call(model, messages, **kwargs)

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        return await asyncio.to_thread(self.completion, model, messages, **kwargs)


class ProviderFactory:
    """Factory to dynamically load provider instances based on the naming conventions."""

    PROVIDERS_DIR = Path(__file__).parent / "providers"

    @classmethod
    def create_provider(cls, provider_key: Union[str, ProviderName], config: ApiConfig) -> Provider:
        """Dynamically load and create an instance of a provider based on the naming convention."""
        # Convert to string if it's a ProviderName enum
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

        # Instantiate the provider class
        provider_class: Type[Provider] = getattr(module, provider_class_name)
        return provider_class(config=config)

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get a list of supported provider keys."""
        # Return the enum values as strings
        return [provider.value for provider in ProviderName]

    @classmethod
    def get_provider_enum(cls, provider_key: str) -> ProviderName:
        """Convert a string provider key to a ProviderName enum."""
        try:
            return ProviderName(provider_key)
        except ValueError:
            supported = [provider.value for provider in ProviderName]
            raise UnsupportedProviderError(provider_key, supported)

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
