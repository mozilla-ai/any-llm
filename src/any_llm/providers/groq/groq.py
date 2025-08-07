from typing import Any, Iterator

try:
    import groq
    from groq import Stream as GroqStream
    from groq.types.chat import ChatCompletionChunk as GroqChatCompletionChunk
    from groq.types.chat import ChatCompletion as GroqChatCompletion
    import instructor
except ImportError:
    msg = "groq or instructor is not installed. Please install it with `pip install any-llm-sdk[groq]`"
    raise ImportError(msg)


from any_llm.types.completion import ChatCompletionChunk, ChatCompletion
from any_llm.provider import Provider, convert_instructor_response
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.helpers import create_completion_from_response
from any_llm.providers.groq.utils import (
    _create_openai_chunk_from_groq_chunk,
    _create_response_dict_from_groq_response,
)


class GroqProvider(Provider):
    """Groq Provider using instructor for structured output."""

    PROVIDER_NAME = "Groq"
    ENV_API_KEY_NAME = "GROQ_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://groq.com/api"

    SUPPORTS_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_REASONING = True
    SUPPORTS_EMBEDDING = False

    @classmethod
    def verify_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Groq provider."""
        if kwargs.get("stream", False) and kwargs.get("response_format", None):
            raise UnsupportedParameterError("stream and response_format", cls.PROVIDER_NAME)

    def _stream_completion(
        self,
        client: groq.Groq,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        stream: GroqStream[GroqChatCompletionChunk] = client.chat.completions.create(  # type: ignore[assignment]
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        for chunk in stream:
            yield _create_openai_chunk_from_groq_chunk(chunk)

    def _make_api_call(
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
            return convert_instructor_response(instructor_response, model, self.PROVIDER_NAME)

        if kwargs.get("stream", False):
            return self._stream_completion(client, model, messages, **kwargs)

        response: GroqChatCompletion = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        response_dict = _create_response_dict_from_groq_response(response)
        return create_completion_from_response(
            response_data=response_dict,
            model=model,
            provider_name=self.PROVIDER_NAME,
        )
