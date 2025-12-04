# Inspired by https://github.com/andrewyng/aisuite/tree/main/aisuite
from __future__ import annotations

import importlib
import os
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from any_llm.constants import INSIDE_NOTEBOOK, LLMProvider
from any_llm.exceptions import MissingApiKeyError, UnsupportedProviderError
from any_llm.tools import prepare_tools
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, CompletionParams
from any_llm.types.provider import PlatformKey, ProviderMetadata
from any_llm.types.responses import Response, ResponseInputParam, ResponsesParams, ResponseStreamEvent
from any_llm.utils.aio import async_iter_to_sync_iter, run_async_in_sync
from any_llm.utils.decorators import BATCH_API_EXPERIMENTAL_MESSAGE, experimental

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence

    from pydantic import BaseModel

    from any_llm.types.batch import Batch
    from any_llm.types.completion import (
        ChatCompletionChunk,
        CreateEmbeddingResponse,
    )
    from any_llm.types.model import Model


class AnyLLM(ABC):
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

    SUPPORTS_BATCH: bool
    """OpenAI Batch Completion API"""

    API_BASE: str | None = None
    """This is used to set the API base for the provider.
    It is not required but may prove useful for providers that have overridable api bases.
    """

    # === Internal Flag Checks ===
    MISSING_PACKAGES_ERROR: ImportError | None = None
    """Some providers use SDKs that are not installed by default.
    This flag is used to check if the packages are installed before instantiating the provider.
    """

    BUILT_IN_TOOLS: ClassVar[list[Any] | None] = None
    """Some providers have built-in tools that can be used as-is without conversion.
    This should be a list of the allowed built-in tool instances.
    For example, in `gemini` provider, this could include `google.genai.types.Tool`.
    """

    ANY_LLM_KEY: str = "ANY_LLM_KEY"

    def __init__(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self._verify_no_missing_packages()
        self._init_client(
            api_key=self._verify_and_set_api_key(api_key),
            api_base=api_base,
            **kwargs,
        )

    def _verify_no_missing_packages(self) -> None:
        if self.MISSING_PACKAGES_ERROR is not None:
            msg = f"{self.PROVIDER_NAME} required packages are not installed. Please install them with `pip install any-llm-sdk[{self.PROVIDER_NAME}]`"
            raise ImportError(msg) from self.MISSING_PACKAGES_ERROR

    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # Standardized API key handling. Splitting into its own function so that providers
        # can easily override this method if they don't want verification (for instance, LMStudio)
        if not api_key:
            api_key = os.getenv(self.ENV_API_KEY_NAME)

        if not api_key:
            raise MissingApiKeyError(self.PROVIDER_NAME, self.ENV_API_KEY_NAME)
        return api_key

    @classmethod
    def create(
        cls, provider: str | LLMProvider, api_key: str | None = None, api_base: str | None = None, **kwargs: Any
    ) -> AnyLLM:
        """Create a provider instance using the given provider name and config.

        Args:
            provider: The provider name (e.g., 'openai', 'anthropic')
            api_key: API key for the provider
            api_base: Base URL for the provider API
            **kwargs: Additional provider-specific arguments

        Returns:
            Provider instance for the specified provider

        """
        return cls._create_provider(provider, api_key=api_key, api_base=api_base, **kwargs)

    @classmethod
    def _create_provider(
        cls, provider_key: str | LLMProvider, api_key: str | None = None, api_base: str | None = None, **kwargs: Any
    ) -> AnyLLM:
        """Dynamically load and create an instance of a provider based on the naming convention."""
        provider_key = LLMProvider.from_string(provider_key).value

        provider_class_name = f"{provider_key.capitalize()}Provider"
        provider_module_name = f"{provider_key}"

        module_path = f"any_llm.providers.{provider_module_name}"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            msg = f"Could not import module {module_path}: {e!s}. Please ensure the provider is supported by doing AnyLLM.get_supported_providers()"
            raise ImportError(msg) from e

        provider_class: type[AnyLLM] = getattr(module, provider_class_name)

        if not api_key:
            api_key = os.getenv(cls.ANY_LLM_KEY)

        if api_key:
            try:
                # Validate if the key conforms with the any-api format.
                # If it does, any-llm must ask any-api for the corresponding provider key.
                PlatformKey(api_key=api_key)

                # Import and instantiate PlatformProvider in-place to avoid circular dependency issues.
                platform_class_name = "PlatformProvider"
                platform_module_path = "any_llm.providers.platform"
                try:
                    platform_module = importlib.import_module(platform_module_path)
                except ImportError as e:
                    msg = f"Could not import module {module_path}: {e!s}. Please ensure the provider is supported by doing AnyLLM.get_supported_providers()"
                    raise ImportError(msg) from e

                platform_class: type[AnyLLM] = getattr(platform_module, platform_class_name)

                # Instantiate the class first and pass the provider next,
                # so we don't change the common API between different provideers.
                platform_provider = platform_class(api_key=api_key, api_base=api_base, **kwargs)
                platform_provider.provider = provider_class  # type: ignore[attr-defined]
            except ValueError:
                pass
            else:
                return platform_provider

        return provider_class(api_key=api_key, api_base=api_base, **kwargs)

    @classmethod
    def get_provider_class(cls, provider_key: str | LLMProvider) -> type[AnyLLM]:
        """Get the provider class without instantiating it.

        Args:
            provider_key: The provider key (e.g., 'anthropic', 'openai')

        Returns:
            The provider class

        """
        provider_key = LLMProvider.from_string(provider_key).value

        provider_class_name = f"{provider_key.capitalize()}Provider"
        provider_module_name = f"{provider_key}"

        module_path = f"any_llm.providers.{provider_module_name}"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            msg = f"Could not import module {module_path}: {e!s}. Please ensure the provider is supported by doing AnyLLM.get_supported_providers()"
            raise ImportError(msg) from e

        provider_class: type[AnyLLM] = getattr(module, provider_class_name)
        return provider_class

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get a list of supported provider keys."""
        return [provider.value for provider in LLMProvider]

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
    def get_provider_enum(cls, provider_key: str) -> LLMProvider:
        """Convert a string provider key to a ProviderName enum."""
        try:
            return LLMProvider(provider_key)
        except ValueError as e:
            supported = [provider.value for provider in LLMProvider]
            raise UnsupportedProviderError(provider_key, supported) from e

    @classmethod
    def split_model_provider(cls, model: str) -> tuple[LLMProvider, str]:
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
            batch_completion=cls.SUPPORTS_BATCH,
            class_name=cls.__name__,
        )

    @abstractmethod
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def completion(
        self,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion synchronously.

        See [AnyLLM.acompletion][any_llm.any_llm.AnyLLM.acompletion]
        """
        allow_running_loop = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        response = run_async_in_sync(self.acompletion(**kwargs), allow_running_loop=allow_running_loop)
        if isinstance(response, ChatCompletion):
            return response

        return async_iter_to_sync_iter(response)

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatCompletionMessage],
        *,
        tools: list[dict[str, Any] | Callable[..., Any]] | Any | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        stream: bool | None = None,
        n: int | None = None,
        stop: str | list[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        user: str | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        logit_bias: dict[str, float] | None = None,
        stream_options: dict[str, Any] | None = None,
        max_completion_tokens: int | None = None,
        reasoning_effort: Literal["minimal", "low", "medium", "high", "auto"] | None = "auto",
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion asynchronously.

        Args:
            model: Model identifier for the chosen provider (e.g., model='gpt-4.1-mini' for LLMProvider.OPENAI).
            messages: List of messages for the conversation
            tools: List of tools for tool calling. Can be Python callables or OpenAI tool format dicts
            tool_choice: Controls which tools the model can call
            temperature: Controls randomness in the response (0.0 to 2.0)
            top_p: Controls diversity via nucleus sampling (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            response_format: Format specification for the response
            stream: Whether to stream the response
            n: Number of completions to generate
            stop: Stop sequences for generation
            presence_penalty: Penalize new tokens based on presence in text
            frequency_penalty: Penalize new tokens based on frequency in text
            seed: Random seed for reproducible results
            user: Unique identifier for the end user
            parallel_tool_calls: Whether to allow parallel tool calls
            logprobs: Include token-level log probabilities in the response
            top_logprobs: Number of alternatives to return when logprobs are requested
            logit_bias: Bias the likelihood of specified tokens during generation
            stream_options: Additional options controlling streaming behavior
            max_completion_tokens: Maximum number of tokens for the completion
            reasoning_effort: Reasoning effort level for models that support it. "auto" will map to each provider's default.
            **kwargs: Additional provider-specific arguments that will be passed to the provider's API call.

        Returns:
            The completion response from the provider

        """
        prepared_tools = None
        if tools:
            prepared_tools = prepare_tools(tools, built_in_tools=self.BUILT_IN_TOOLS)

        processed_messages: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, ChatCompletionMessage):
                # Dump the message but exclude the extra field that we extend from OpenAI Spec
                processed_messages.append(message.model_dump(exclude_none=True, exclude={"reasoning"}))
            else:
                processed_messages.append(message)

        params = CompletionParams(
            model_id=model,
            messages=processed_messages,
            tools=prepared_tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=response_format,
            stream=stream,
            n=n,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            user=user,
            parallel_tool_calls=parallel_tool_calls,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            logit_bias=logit_bias,
            stream_options=stream_options,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )

        return await self._acompletion(params, **kwargs)

    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if not self.SUPPORTS_COMPLETION:
            msg = "Provider doesn't support completion."
            raise NotImplementedError(msg)
        msg = "Subclasses must implement _acompletion method"
        raise NotImplementedError(msg)

    def responses(self, **kwargs: Any) -> Response | Iterator[ResponseStreamEvent]:
        """Create a response synchronously.

        See [AnyLLM.aresponses][any_llm.any_llm.AnyLLM.aresponses]
        """
        allow_running_loop = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        response = run_async_in_sync(self.aresponses(**kwargs), allow_running_loop=allow_running_loop)
        if isinstance(response, Response):
            return response
        return async_iter_to_sync_iter(response)

    async def aresponses(
        self,
        model: str,
        input_data: str | ResponseInputParam,
        *,
        tools: list[dict[str, Any] | Callable[..., Any]] | Any | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool | None = None,
        instructions: str | None = None,
        max_tool_calls: int | None = None,
        parallel_tool_calls: int | None = None,
        reasoning: Any | None = None,
        text: Any | None = None,
        **kwargs: Any,
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """Create a response using the OpenAI-style Responses API.

        This follows the OpenAI Responses API shape and returns the aliased
        `any_llm.types.responses.Response` type. If `stream=True`, an iterator of
        `any_llm.types.responses.ResponseStreamEvent` items is returned.

        Args:
            model: Model identifier for the chosen provider (e.g., model='gpt-4.1-mini' for LLMProvider.OPENAI).
            input_data: The input payload accepted by provider's Responses API.
                For OpenAI-compatible providers, this is typically a list mixing
                text, images, and tool instructions, or a dict per OpenAI spec.
            tools: Optional tools for tool calling (Python callables or OpenAI tool dicts)
            tool_choice: Controls which tools the model can call
            max_output_tokens: Maximum number of output tokens to generate
            temperature: Controls randomness in the response (0.0 to 2.0)
            top_p: Controls diversity via nucleus sampling (0.0 to 1.0)
            stream: Whether to stream response events
            instructions: A system (or developer) message inserted into the model's context.
            max_tool_calls: The maximum number of total calls to built-in tools that can be processed in a response. This maximum number applies across all built-in tool calls, not per individual tool. Any further attempts to call a tool by the model will be ignored.
            parallel_tool_calls: Whether to allow the model to run tool calls in parallel.
            reasoning: Configuration options for reasoning models.
            text: Configuration options for a text response from the model. Can be plain text or structured JSON data.
            **kwargs: Additional provider-specific arguments that will be passed to the provider's API call.

        Returns:
            Either a `Response` object (non-streaming) or an iterator of
            `ResponseStreamEvent` (streaming).

        Raises:
            NotImplementedError: If the selected provider does not support the Responses API.

        """
        prepared_tools = None
        if tools:
            prepared_tools = prepare_tools(tools, built_in_tools=self.BUILT_IN_TOOLS)

        params = ResponsesParams(
            model=model,
            input=input_data,
            tools=prepared_tools,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            instructions=instructions,
            max_tool_calls=max_tool_calls,
            parallel_tool_calls=bool(parallel_tool_calls),
            reasoning=reasoning,
            text=text,
            **kwargs,
        )

        return await self._aresponses(params)

    async def _aresponses(
        self, params: ResponsesParams, **kwargs: Any
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        if not self.SUPPORTS_RESPONSES:
            msg = "Provider doesn't support responses."
            raise NotImplementedError(msg)
        msg = "Subclasses must implement _aresponses method"
        raise NotImplementedError(msg)

    def _embedding(self, model: str, inputs: str | list[str], **kwargs: Any) -> CreateEmbeddingResponse:
        allow_running_loop = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(self.aembedding(model, inputs, **kwargs), allow_running_loop=allow_running_loop)

    async def aembedding(self, model: str, inputs: str | list[str], **kwargs: Any) -> CreateEmbeddingResponse:
        return await self._aembedding(model, inputs, **kwargs)

    async def _aembedding(self, model: str, inputs: str | list[str], **kwargs: Any) -> CreateEmbeddingResponse:
        if not self.SUPPORTS_EMBEDDING:
            msg = "Provider doesn't support embedding."
            raise NotImplementedError(msg)
        msg = "Subclasses must implement _aembedding method"
        raise NotImplementedError(msg)

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        allow_running_loop = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(self.alist_models(**kwargs), allow_running_loop=allow_running_loop)

    async def alist_models(self, **kwargs: Any) -> Sequence[Model]:
        return await self._alist_models(**kwargs)

    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        if not self.SUPPORTS_LIST_MODELS:
            msg = "Provider doesn't support listing models."
            raise NotImplementedError(msg)
        msg = "Subclasses must implement _alist_models method"
        raise NotImplementedError(msg)

    @experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
    def create_batch(self, **kwargs: Any) -> Batch:
        """Create a batch synchronously.

        See [AnyLLM.acreate_batch][any_llm.any_llm.AnyLLM.acreate_batch]
        """
        allow_running_loop = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(self.acreate_batch(**kwargs), allow_running_loop=allow_running_loop)

    @experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
    async def acreate_batch(
        self,
        input_file_path: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Create a batch job asynchronously.

        Args:
            input_file_path: Path to a local file containing batch requests in JSONL format.
            endpoint: The endpoint to be used for all requests (e.g., '/v1/chat/completions')
            completion_window: The time frame within which the batch should be processed (default: '24h')
            metadata: Optional custom metadata for the batch
            **kwargs: Additional provider-specific arguments

        Returns:
            The created batch object

        """
        return await self._acreate_batch(
            input_file_path=input_file_path,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
            **kwargs,
        )

    async def _acreate_batch(
        self,
        input_file_path: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Batch:
        if not self.SUPPORTS_BATCH:
            msg = "Provider doesn't support batch completions."
            raise NotImplementedError(msg)
        msg = "Subclasses must implement _acreate_batch method"
        raise NotImplementedError(msg)

    @experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
    def retrieve_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Retrieve a batch synchronously.

        See [AnyLLM.aretrieve_batch][any_llm.any_llm.AnyLLM.aretrieve_batch]
        """
        allow_running_loop = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(self.aretrieve_batch(batch_id, **kwargs), allow_running_loop=allow_running_loop)

    @experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
    async def aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Retrieve a batch job asynchronously.

        Args:
            batch_id: The ID of the batch to retrieve
            **kwargs: Additional provider-specific arguments

        Returns:
            The batch object

        """
        return await self._aretrieve_batch(batch_id, **kwargs)

    async def _aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        if not self.SUPPORTS_BATCH:
            msg = "Provider doesn't support batch completions."
            raise NotImplementedError(msg)
        msg = "Subclasses must implement _aretrieve_batch method"
        raise NotImplementedError(msg)

    @experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
    def cancel_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Cancel a batch synchronously.

        See [AnyLLM.acancel_batch][any_llm.any_llm.AnyLLM.acancel_batch]
        """
        allow_running_loop = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(self.acancel_batch(batch_id, **kwargs), allow_running_loop=allow_running_loop)

    @experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
    async def acancel_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Cancel a batch job asynchronously.

        Args:
            batch_id: The ID of the batch to cancel
            **kwargs: Additional provider-specific arguments

        Returns:
            The cancelled batch object

        """
        return await self._acancel_batch(batch_id, **kwargs)

    async def _acancel_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        if not self.SUPPORTS_BATCH:
            msg = "Provider doesn't support batch completions."
            raise NotImplementedError(msg)
        msg = "Subclasses must implement _acancel_batch method"
        raise NotImplementedError(msg)

    @experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
    def list_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Sequence[Batch]:
        """List batches synchronously.

        See [AnyLLM.alist_batches][any_llm.any_llm.AnyLLM.alist_batches]
        """
        allow_running_loop = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(
            self.alist_batches(after=after, limit=limit, **kwargs), allow_running_loop=allow_running_loop
        )

    @experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
    async def alist_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Sequence[Batch]:
        """List batch jobs asynchronously.

        Args:
            after: A cursor for pagination. Returns batches after this batch ID.
            limit: Maximum number of batches to return (default: 20)
            **kwargs: Additional provider-specific arguments

        Returns:
            A list of batch objects

        """
        return await self._alist_batches(after=after, limit=limit, **kwargs)

    async def _alist_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Sequence[Batch]:
        if not self.SUPPORTS_BATCH:
            msg = "Provider doesn't support batch completions."
            raise NotImplementedError(msg)
        msg = "Subclasses must implement _alist_batches method"
        raise NotImplementedError(msg)
