import os
from collections.abc import AsyncIterator, Iterable
from typing import Any

import openai
from openai import AsyncOpenAI

from any_llm.any_llm import AnyLLM
from any_llm.errors import (
    AnyLLMError,
    AuthenticationError,
    ContentFilterError,
    ContextLengthExceededError,
    InvalidRequestError,
    MissingApiKeyError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from any_llm.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    CompletionParams,
)

PROVIDER_NAME = "OpenAI"
ENV_VAR_NAME = "OPENAI_API_KEY"


class OpenAIProvider(AnyLLM):
    """OpenAI provider implementation using the official OpenAI SDK."""

    _client: AsyncOpenAI

    def _init_client(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI async client.

        Args:
            api_key: API key for OpenAI. If None, checks OPENAI_API_KEY env var.
            api_base: Base URL for the API. If None, uses OpenAI default.
            **kwargs: Additional arguments passed to AsyncOpenAI.

        Raises:
            MissingApiKeyError: If no API key is provided or found in environment.

        """
        resolved_key = api_key or os.environ.get(ENV_VAR_NAME)
        if not resolved_key:
            raise MissingApiKeyError(PROVIDER_NAME, ENV_VAR_NAME)

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=api_base,
            **kwargs,
        )

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
        """Perform a chat completion request using OpenAI.

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
            **kwargs: Additional arguments passed to the API.

        Returns:
            ChatCompletion response object.

        """
        params = self._convert_completion_params(
            CompletionParams(
                model=model,
                messages=list(messages),
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
                tools=list(tools) if tools else None,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
            ),
            **kwargs,
        )

        try:
            response = await self._client.chat.completions.create(**params)
            return self._convert_completion_response(response)
        except openai.APIError as e:
            raise self._map_openai_error(e) from e

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
        """Perform a streaming chat completion request using OpenAI.

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
            **kwargs: Additional arguments passed to the API.

        Yields:
            ChatCompletionChunk objects.

        """
        params = self._convert_completion_params(
            CompletionParams(
                model=model,
                messages=list(messages),
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
                stream=True,
                stream_options=stream_options,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=list(tools) if tools else None,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
            ),
            **kwargs,
        )

        try:
            stream = await self._client.chat.completions.create(**params)
            async for chunk in stream:
                yield self._convert_completion_chunk_response(chunk)
        except openai.APIError as e:
            raise self._map_openai_error(e) from e

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert completion parameters to OpenAI API format.

        For OpenAI, this is a direct mapping since OpenAI is the canonical format.

        Args:
            params: The completion parameters.
            **kwargs: Additional arguments to include.

        Returns:
            OpenAI API-compatible parameters dictionary.

        """
        api_params = params.to_api_params()
        api_params.update(kwargs)
        return api_params

    @staticmethod
    def _convert_completion_response(response: ChatCompletion) -> ChatCompletion:
        """Convert OpenAI response to ChatCompletion format.

        For OpenAI, this is a pass-through since the response is already in the correct format.

        Args:
            response: The OpenAI API response.

        Returns:
            ChatCompletion object.

        """
        # The OpenAI SDK returns ChatCompletion objects directly
        return response

    @staticmethod
    def _convert_completion_chunk_response(response: ChatCompletionChunk, **kwargs: Any) -> ChatCompletionChunk:
        """Convert OpenAI streaming chunk to ChatCompletionChunk format.

        For OpenAI, this is a pass-through since the chunk is already in the correct format.

        Args:
            response: The OpenAI streaming chunk.
            **kwargs: Additional arguments (unused).

        Returns:
            ChatCompletionChunk object.

        """
        # The OpenAI SDK returns ChatCompletionChunk objects directly
        return response

    @staticmethod
    def _map_openai_error(error: openai.APIError) -> AnyLLMError:
        """Map OpenAI SDK errors to AnyLLM error types.

        Args:
            error: The OpenAI API error.

        Returns:
            Corresponding AnyLLMError subclass.

        """
        error_message = str(error)

        if isinstance(error, openai.AuthenticationError):
            return AuthenticationError(
                message=error_message,
                original_exception=error,
                provider_name=PROVIDER_NAME,
            )

        if isinstance(error, openai.RateLimitError):
            return RateLimitError(
                message=error_message,
                original_exception=error,
                provider_name=PROVIDER_NAME,
            )

        if isinstance(error, openai.BadRequestError):
            # Check for specific error types based on message content
            lower_message = error_message.lower()
            if "context length" in lower_message or "maximum context" in lower_message:
                return ContextLengthExceededError(
                    message=error_message,
                    original_exception=error,
                    provider_name=PROVIDER_NAME,
                )
            if "content filter" in lower_message or "content_filter" in lower_message:
                return ContentFilterError(
                    message=error_message,
                    original_exception=error,
                    provider_name=PROVIDER_NAME,
                )
            return InvalidRequestError(
                message=error_message,
                original_exception=error,
                provider_name=PROVIDER_NAME,
            )

        if isinstance(error, openai.NotFoundError):
            return ModelNotFoundError(
                message=error_message,
                original_exception=error,
                provider_name=PROVIDER_NAME,
            )

        if isinstance(error, openai.InternalServerError):
            return ProviderError(
                message=error_message,
                original_exception=error,
                provider_name=PROVIDER_NAME,
            )

        # Default to generic provider error
        return ProviderError(
            message=error_message,
            original_exception=error,
            provider_name=PROVIDER_NAME,
        )
