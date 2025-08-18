from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, cast

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.types.responses import Response, ResponseStreamEvent

try:
    import groq
    import instructor
    from groq import AsyncStream as GroqAsyncStream
    from groq import Stream as GroqStream

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False


from any_llm.provider import Provider
from any_llm.providers.groq.utils import (
    _create_openai_chunk_from_groq_chunk,
    to_chat_completion,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.utils.instructor import _convert_instructor_response

if TYPE_CHECKING:
    from groq.types.chat import ChatCompletion as GroqChatCompletion
    from groq.types.chat import ChatCompletionChunk as GroqChatCompletionChunk


class GroqProvider(Provider):
    """Groq Provider using instructor for structured output."""

    PROVIDER_NAME = "groq"
    ENV_API_KEY_NAME = "GROQ_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://groq.com/api"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = False

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    async def _stream_async_completion(
        self, client: groq.AsyncGroq, params: CompletionParams, **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        if params.stream and params.response_format:
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)

        stream: GroqAsyncStream[GroqChatCompletionChunk] = await client.chat.completions.create(
            model=params.model_id,
            messages=cast("Any", params.messages),
            **params.model_dump(exclude_none=True, exclude={"model_id", "messages"}),
            **kwargs,
        )

        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in stream:
                yield _create_openai_chunk_from_groq_chunk(chunk)

        return _stream()

    def _stream_completion(
        self,
        client: groq.Groq,
        params: CompletionParams,
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        if params.stream and params.response_format:
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        stream: GroqStream[GroqChatCompletionChunk] = client.chat.completions.create(
            model=params.model_id,
            messages=cast("Any", params.messages),
            **params.model_dump(exclude_none=True, exclude={"model_id", "messages"}),
            **kwargs,
        )
        for chunk in stream:
            yield _create_openai_chunk_from_groq_chunk(chunk)

    async def acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using Groq."""
        client = groq.AsyncGroq(api_key=self.config.api_key)

        if params.response_format:
            instructor_client = instructor.from_groq(client, mode=instructor.Mode.JSON)
            if not isinstance(params.response_format, type) or not issubclass(params.response_format, BaseModel):
                msg = "response_format must be a pydantic model"
                raise ValueError(msg)
            instructor_response = await instructor_client.chat.completions.create(
                model=params.model_id,
                messages=params.messages,  # type: ignore[arg-type]
                response_model=params.response_format,
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "stream"}),
                **kwargs,
            )
            return _convert_instructor_response(instructor_response, params.model_id, self.PROVIDER_NAME)

        if params.stream:
            return await self._stream_async_completion(
                client,
                params,
                **kwargs,
            )
        response: GroqChatCompletion = await client.chat.completions.create(
            model=params.model_id,
            messages=cast("Any", params.messages),
            **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "stream"}),
            **kwargs,
        )

        return to_chat_completion(response)

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Groq."""
        client = groq.Groq(api_key=self.config.api_key)

        if params.response_format:
            instructor_client = instructor.from_groq(client, mode=instructor.Mode.JSON)
            if not isinstance(params.response_format, type) or not issubclass(params.response_format, BaseModel):
                msg = "response_format must be a pydantic model"
                raise ValueError(msg)
            instructor_response = instructor_client.chat.completions.create(
                model=params.model_id,
                messages=params.messages,  # type: ignore[arg-type]
                response_model=params.response_format,
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "stream"}),
                **kwargs,
            )
            return _convert_instructor_response(instructor_response, params.model_id, self.PROVIDER_NAME)

        if params.stream:
            return self._stream_completion(
                client,
                params,
                **kwargs,
            )
        response: GroqChatCompletion = client.chat.completions.create(
            model=params.model_id,
            messages=cast("Any", params.messages),
            **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "stream"}),
            **kwargs,
        )

        return to_chat_completion(response)

    async def aresponses(
        self, model: str, input_data: Any, **kwargs: Any
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """Call Groq Responses API and normalize into ChatCompletion/Chunks."""
        # Python SDK doesn't yet support it: https://community.groq.com/feature-requests-6/groq-python-sdk-support-for-responses-api-262

        if "max_tool_calls" in kwargs:
            parameter = "max_tool_calls"
            raise UnsupportedParameterError(parameter, self.PROVIDER_NAME)

        client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.config.api_key,
        )

        response = await client.responses.create(
            model=model,
            input=input_data,
            **kwargs,
        )

        if not isinstance(response, Response | AsyncStream):
            msg = f"Responses API returned an unexpected type: {type(response)}"
            raise ValueError(msg)

        return response

    def responses(self, model: str, input_data: Any, **kwargs: Any) -> Response | Iterator[ResponseStreamEvent]:
        """Call Groq Responses API and normalize into ChatCompletion/Chunks."""
        # Python SDK doesn't yet support it: https://community.groq.com/feature-requests-6/groq-python-sdk-support-for-responses-api-262

        if "max_tool_calls" in kwargs:
            parameter = "max_tool_calls"
            raise UnsupportedParameterError(parameter, self.PROVIDER_NAME)

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.config.api_key,
        )
        response = client.responses.create(
            model=model,
            input=input_data,
            **kwargs,
        )
        if not isinstance(response, Response | Stream):
            msg = f"Responses API returned an unexpected type: {type(response)}"
            raise ValueError(msg)
        return response
