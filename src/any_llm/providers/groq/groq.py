from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from openai import OpenAI, Stream

from any_llm.types.responses import Response, ResponseStreamEvent

try:
    import groq
    import instructor
    from groq import Stream as GroqStream
except ImportError as exc:
    msg = "groq or instructor is not installed. Please install it with `pip install any-llm-sdk[groq]`"
    raise ImportError(msg) from exc


from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import Provider
from any_llm.providers.groq.utils import (
    _create_openai_chunk_from_groq_chunk,
    to_chat_completion,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
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

    def _stream_completion(
        self,
        client: groq.Groq,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        if kwargs.get("stream", False) and kwargs.get("response_format", None):
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        stream: GroqStream[GroqChatCompletionChunk] = client.chat.completions.create(  # type: ignore[assignment]
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        for chunk in stream:
            yield _create_openai_chunk_from_groq_chunk(chunk)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Groq."""
        client = groq.Groq(api_key=self.config.api_key)

        if "response_format" in kwargs:
            instructor_client = instructor.from_groq(client, mode=instructor.Mode.JSON)
            response_format = kwargs.pop("response_format")
            instructor_response = instructor_client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_model=response_format,
                **kwargs,
            )
            return _convert_instructor_response(instructor_response, model, self.PROVIDER_NAME)

        if kwargs.get("stream", False):
            return self._stream_completion(client, model, messages, **kwargs)

        response: GroqChatCompletion = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        return to_chat_completion(response)

    def responses(self, model: str, input_data: Any, **kwargs: Any) -> Response | Iterator[ResponseStreamEvent]:
        """Call Groq Responses API and normalize into ChatCompletion/Chunks."""
        # Python SDK doesn't yet support it: https://community.groq.com/feature-requests-6/groq-python-sdk-support-for-responses-api-262
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
