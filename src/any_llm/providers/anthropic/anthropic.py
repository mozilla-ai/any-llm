from collections.abc import AsyncIterator, Iterator
from typing import Any

from pydantic import BaseModel

try:
    import instructor
    from anthropic import Anthropic, AsyncAnthropic

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from any_llm.exceptions import UnsupportedParameterError
from any_llm.logging import logger
from any_llm.provider import Provider
from any_llm.providers.anthropic.utils import (
    DEFAULT_MAX_TOKENS,
    _convert_messages_for_anthropic,
    _convert_response,
    _convert_tool_choice,
    _convert_tool_spec,
    _create_openai_chunk_from_anthropic_chunk,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.utils.instructor import _convert_instructor_response


class AnthropicProvider(Provider):
    """
    Anthropic Provider using enhanced Provider framework.

    Handles conversion between OpenAI format and Anthropic's native format.
    """

    PROVIDER_NAME = "anthropic"
    ENV_API_KEY_NAME = "ANTHROPIC_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.anthropic.com/en/home"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    async def _stream_completion_async(
        self, client: "AsyncAnthropic", model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        if kwargs.get("response_format", None):
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        """Handle streaming completion - extracted to avoid generator issues."""
        system_message, filtered_messages = _convert_messages_for_anthropic(messages)

        # Prepare kwargs for Anthropic
        anthropic_kwargs = kwargs.copy()
        if system_message:
            anthropic_kwargs["system"] = system_message

        async with client.messages.stream(
            model=model,
            messages=filtered_messages,  # type: ignore[arg-type]
            **anthropic_kwargs,
        ) as anthropic_stream:
            async for event in anthropic_stream:
                yield _create_openai_chunk_from_anthropic_chunk(event)

    def _stream_completion(
        self,
        client: "Anthropic",
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        if kwargs.get("response_format", None):
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        """Handle streaming completion - extracted to avoid generator issues."""
        system_message, filtered_messages = _convert_messages_for_anthropic(messages)

        # Prepare kwargs for Anthropic
        anthropic_kwargs = kwargs.copy()
        if system_message:
            anthropic_kwargs["system"] = system_message

        with client.messages.stream(
            model=model,
            messages=filtered_messages,  # type: ignore[arg-type]
            **anthropic_kwargs,
        ) as anthropic_stream:
            for event in anthropic_stream:
                yield _create_openai_chunk_from_anthropic_chunk(event)

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using Anthropic with instructor support."""
        client = AsyncAnthropic(api_key=self.config.api_key, base_url=self.config.api_base)

        if params.max_tokens is None:
            logger.warning(f"max_tokens is required for Anthropic, setting to {DEFAULT_MAX_TOKENS}")
            params.max_tokens = DEFAULT_MAX_TOKENS

        if params.tools:
            params.tools = _convert_tool_spec(params.tools)

        if params.tool_choice or params.parallel_tool_calls:
            params.tool_choice = _convert_tool_choice(params)

        params_kwargs = params.model_dump(
            exclude_none=True, exclude={"model_id", "messages", "response_format", "parallel_tool_calls"}
        )
        if params.response_format:
            instructor_client = instructor.from_anthropic(client)

            response_format = params.response_format

            if not isinstance(response_format, type) or not issubclass(response_format, BaseModel):
                msg = "Instructor response_format must be a pydantic model"
                raise ValueError(msg)

            system_message, filtered_messages = _convert_messages_for_anthropic(params.messages)

            instructor_kwargs = kwargs.copy()
            if system_message:
                instructor_kwargs["system"] = system_message

            instructor_response = await instructor_client.messages.create(
                model=params.model_id,
                messages=filtered_messages,  # type: ignore[arg-type]
                response_model=response_format,
                **instructor_kwargs,
                **params_kwargs,
            )

            return _convert_instructor_response(instructor_response, params.model_id, self.PROVIDER_NAME)

        if params.stream:
            params_kwargs.pop("stream")
            return self._stream_completion_async(client, params.model_id, params.messages, **params_kwargs)

        system_message, filtered_messages = _convert_messages_for_anthropic(params.messages)

        anthropic_kwargs = kwargs.copy()
        if system_message:
            anthropic_kwargs["system"] = system_message

        message = await client.messages.create(
            model=params.model_id,
            messages=filtered_messages,  # type: ignore[arg-type]
            **anthropic_kwargs,
            **params_kwargs,
        )
        return _convert_response(message)

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Anthropic with instructor support."""

        client = Anthropic(api_key=self.config.api_key, base_url=self.config.api_base)

        if params.max_tokens is None:
            logger.warning(f"max_tokens is required for Anthropic, setting to {DEFAULT_MAX_TOKENS}")
            params.max_tokens = DEFAULT_MAX_TOKENS

        if params.tools:
            params.tools = _convert_tool_spec(params.tools)

        if params.tool_choice or params.parallel_tool_calls:
            params.tool_choice = _convert_tool_choice(params)

        params_kwargs = params.model_dump(
            exclude_none=True, exclude={"model_id", "messages", "response_format", "parallel_tool_calls"}
        )
        if params.response_format:
            instructor_client = instructor.from_anthropic(client)

            response_format = params.response_format

            if not isinstance(response_format, type) or not issubclass(response_format, BaseModel):
                msg = "Instructor response_format must be a pydantic model"
                raise ValueError(msg)

            system_message, filtered_messages = _convert_messages_for_anthropic(params.messages)

            instructor_kwargs = kwargs.copy()
            if system_message:
                instructor_kwargs["system"] = system_message

            instructor_response = instructor_client.messages.create(
                model=params.model_id,
                messages=filtered_messages,  # type: ignore[arg-type]
                response_model=response_format,
                **instructor_kwargs,
                **params_kwargs,
            )

            return _convert_instructor_response(instructor_response, params.model_id, self.PROVIDER_NAME)

        if params.stream:
            params_kwargs.pop("stream")
            return self._stream_completion(client, params.model_id, params.messages, **params_kwargs)
        system_message, filtered_messages = _convert_messages_for_anthropic(params.messages)

        anthropic_kwargs = kwargs.copy()
        if system_message:
            anthropic_kwargs["system"] = system_message

        message = client.messages.create(
            model=params.model_id,
            messages=filtered_messages,  # type: ignore[arg-type]
            **anthropic_kwargs,
            **params_kwargs,
        )
        return _convert_response(message)
