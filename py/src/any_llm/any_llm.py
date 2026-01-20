from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

from any_llm.providers import Providers, get_provider_class
from any_llm.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    CompletionParams,
)
from any_llm.utils.aio import async_iter_to_sync_iter, run_async_in_sync


class AnyLLM(ABC):
    """Abstract base class for LLM providers.

    All providers must inherit from this class and implement the abstract methods.
    """

    @classmethod
    async def acreate(
        cls,
        provider: str | Providers,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> "AnyLLM":
        """Create a provider-specific LLM client asynchronously.

        Args:
            provider: The provider name or Providers enum value.
            api_key: API key for the provider. If None, will check environment variables.
            api_base: Base URL for the API. If None, uses provider default.
            **kwargs: Additional provider-specific arguments.

        Returns:
            An initialized LLM client for the specified provider.

        """
        provider_class = get_provider_class(provider)
        instance: AnyLLM = provider_class()
        instance._init_client(api_key=api_key, api_base=api_base, **kwargs)
        return instance

    @classmethod
    def create(
        cls,
        provider: str | Providers,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> "AnyLLM":
        """Create a provider-specific LLM client synchronously.

        Args:
            provider: The provider name or Providers enum value.
            api_key: API key for the provider. If None, will check environment variables.
            api_base: Base URL for the API. If None, uses provider default.
            **kwargs: Additional provider-specific arguments.

        Returns:
            An initialized LLM client for the specified provider.

        """
        return run_async_in_sync(cls.acreate(provider, api_key, api_base, **kwargs))

    @abstractmethod
    def _init_client(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HTTP/Provider SDK client.

        Args:
            api_key: API key for the provider. If None, should check environment variables.
            api_base: Base URL for the API. If None, uses provider default.
            **kwargs: Additional provider-specific arguments.

        """

    @abstractmethod
    async def acompletion(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        *,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Perform a chat completion request asynchronously.

        Args:
            model: Model ID to use for the completion.
            messages: List of messages comprising the conversation.
            frequency_penalty: Penalize repeated tokens (-2.0 to 2.0).
            logit_bias: Map of token IDs to bias values.
            logprobs: Whether to return log probabilities.
            max_completion_tokens: Maximum tokens to generate.
            max_tokens: Legacy parameter for maximum tokens.
            n: Number of completions to generate.
            parallel_tool_calls: Enable parallel function calling.
            presence_penalty: Penalize new topics (-2.0 to 2.0).
            response_format: Format specification for structured outputs.
            seed: Seed for deterministic sampling.
            stop: Stop sequences.
            stream_options: Options for streaming.
            temperature: Sampling temperature (0 to 2).
            tool_choice: Controls tool calling behavior.
            tools: List of tools the model may call.
            top_logprobs: Number of top tokens to return.
            top_p: Nucleus sampling threshold.
            user: Unique user identifier.
            **kwargs: Additional provider-specific arguments.

        Returns:
            ChatCompletion response object.

        """

    def completion(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        *,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Perform a chat completion request synchronously.

        Args:
            model: Model ID to use for the completion.
            messages: List of messages comprising the conversation.
            frequency_penalty: Penalize repeated tokens (-2.0 to 2.0).
            logit_bias: Map of token IDs to bias values.
            logprobs: Whether to return log probabilities.
            max_completion_tokens: Maximum tokens to generate.
            max_tokens: Legacy parameter for maximum tokens.
            n: Number of completions to generate.
            parallel_tool_calls: Enable parallel function calling.
            presence_penalty: Penalize new topics (-2.0 to 2.0).
            response_format: Format specification for structured outputs.
            seed: Seed for deterministic sampling.
            stop: Stop sequences.
            stream_options: Options for streaming.
            temperature: Sampling temperature (0 to 2).
            tool_choice: Controls tool calling behavior.
            tools: List of tools the model may call.
            top_logprobs: Number of top tokens to return.
            top_p: Nucleus sampling threshold.
            user: Unique user identifier.
            **kwargs: Additional provider-specific arguments.

        Returns:
            ChatCompletion response object.

        """
        return run_async_in_sync(
            self.acompletion(
                model=model,
                messages=messages,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=logprobs,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                n=n,
                parallel_tool_calls=parallel_tool_calls,
                presence_penalty=presence_penalty,
                response_format=response_format,
                seed=seed,
                stop=stop,
                stream_options=stream_options,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=tools,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
                **kwargs,
            )
        )

    @abstractmethod
    async def acompletion_stream(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        *,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Perform a streaming chat completion request asynchronously.

        Args:
            model: Model ID to use for the completion.
            messages: List of messages comprising the conversation.
            frequency_penalty: Penalize repeated tokens (-2.0 to 2.0).
            logit_bias: Map of token IDs to bias values.
            logprobs: Whether to return log probabilities.
            max_completion_tokens: Maximum tokens to generate.
            max_tokens: Legacy parameter for maximum tokens.
            n: Number of completions to generate.
            parallel_tool_calls: Enable parallel function calling.
            presence_penalty: Penalize new topics (-2.0 to 2.0).
            response_format: Format specification for structured outputs.
            seed: Seed for deterministic sampling.
            stop: Stop sequences.
            stream_options: Options for streaming.
            temperature: Sampling temperature (0 to 2).
            tool_choice: Controls tool calling behavior.
            tools: List of tools the model may call.
            top_logprobs: Number of top tokens to return.
            top_p: Nucleus sampling threshold.
            user: Unique user identifier.
            **kwargs: Additional provider-specific arguments.

        Yields:
            ChatCompletionChunk objects.

        """
        # This yield is required for abstract async generators
        if False:
            yield ChatCompletionChunk(
                id="",
                choices=[],
                created=0,
                model="",
                object="chat.completion.chunk",
            )

    def completion_stream(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        *,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Perform a streaming chat completion request synchronously.

        Args:
            model: Model ID to use for the completion.
            messages: List of messages comprising the conversation.
            frequency_penalty: Penalize repeated tokens (-2.0 to 2.0).
            logit_bias: Map of token IDs to bias values.
            logprobs: Whether to return log probabilities.
            max_completion_tokens: Maximum tokens to generate.
            max_tokens: Legacy parameter for maximum tokens.
            n: Number of completions to generate.
            parallel_tool_calls: Enable parallel function calling.
            presence_penalty: Penalize new topics (-2.0 to 2.0).
            response_format: Format specification for structured outputs.
            seed: Seed for deterministic sampling.
            stop: Stop sequences.
            stream_options: Options for streaming.
            temperature: Sampling temperature (0 to 2).
            tool_choice: Controls tool calling behavior.
            tools: List of tools the model may call.
            top_logprobs: Number of top tokens to return.
            top_p: Nucleus sampling threshold.
            user: Unique user identifier.
            **kwargs: Additional provider-specific arguments.

        Yields:
            ChatCompletionChunk objects.

        """
        async_iter = self.acompletion_stream(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream_options=stream_options,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
            **kwargs,
        )
        return async_iter_to_sync_iter(async_iter)

    @staticmethod
    @abstractmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert completion parameters to provider-specific format.

        Args:
            params: The completion parameters.
            **kwargs: Additional arguments.

        Returns:
            Provider-specific parameters dictionary.

        """

    @staticmethod
    @abstractmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert provider response to ChatCompletion format.

        Args:
            response: The provider-specific response.

        Returns:
            ChatCompletion object.

        """

    @staticmethod
    @abstractmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert provider streaming response chunk to ChatCompletionChunk format.

        Args:
            response: The provider-specific chunk response.
            **kwargs: Additional arguments.

        Returns:
            ChatCompletionChunk object.

        """
