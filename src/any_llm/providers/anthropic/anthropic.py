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
from any_llm.provider import Provider
from any_llm.providers.anthropic.utils import (
    _convert_params,
    _convert_response,
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
    SUPPORTS_LIST_MODELS = False

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    async def _stream_completion_async(
        self, client: "AsyncAnthropic", **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        if kwargs.get("response_format", None):
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        """Handle streaming completion - extracted to avoid generator issues."""

        async with client.messages.stream(
            **kwargs,
        ) as anthropic_stream:
            async for event in anthropic_stream:
                yield _create_openai_chunk_from_anthropic_chunk(event)

    def _stream_completion(
        self,
        client: "Anthropic",
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        if kwargs.get("response_format", None):
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        """Handle streaming completion - extracted to avoid generator issues."""

        with client.messages.stream(
            **kwargs,
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

        converted_kwargs = _convert_params(params, **kwargs)

        if params.response_format:
            instructor_client = instructor.from_anthropic(client)

            response_format = params.response_format

            if not isinstance(response_format, type) or not issubclass(response_format, BaseModel):
                msg = "Instructor response_format must be a pydantic model"
                raise ValueError(msg)

            instructor_response = await instructor_client.messages.create(
                **converted_kwargs,
                response_model=response_format,
            )

            return _convert_instructor_response(instructor_response, params.model_id, self.PROVIDER_NAME)

        if converted_kwargs.pop("stream", False):
            return self._stream_completion_async(client, **converted_kwargs)

        message = await client.messages.create(**converted_kwargs)

        return _convert_response(message)

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Anthropic with instructor support."""

        client = Anthropic(api_key=self.config.api_key, base_url=self.config.api_base)

        converted_kwargs = _convert_params(params, **kwargs)

        if params.response_format:
            instructor_client = instructor.from_anthropic(client)

            response_format = params.response_format

            if not isinstance(response_format, type) or not issubclass(response_format, BaseModel):
                msg = "Instructor response_format must be a pydantic model"
                raise ValueError(msg)

            instructor_response = instructor_client.messages.create(
                **converted_kwargs,
                response_model=response_format,
            )

            return _convert_instructor_response(instructor_response, params.model_id, self.PROVIDER_NAME)

        if converted_kwargs.pop("stream", False):
            return self._stream_completion(client, **converted_kwargs)

        message = client.messages.create(**converted_kwargs)

        return _convert_response(message)
