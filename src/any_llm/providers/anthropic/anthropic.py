from typing import Any, Iterator


try:
    from anthropic import Anthropic
    import instructor
except ImportError:
    msg = "anthropic or instructor is not installed. Please install it with `pip install any-llm-sdk[anthropic]`"
    raise ImportError(msg)

from any_llm.types.completion import ChatCompletion
from any_llm.types.completion import ChatCompletionChunk

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import Provider, convert_instructor_response
from any_llm.providers.anthropic.utils import (
    _create_openai_chunk_from_anthropic_chunk,
    _convert_response,
    _convert_kwargs,
    _convert_messages_for_anthropic,
)


class AnthropicProvider(Provider):
    """
    Anthropic Provider using enhanced Provider framework.

    Handles conversion between OpenAI format and Anthropic's native format.
    """

    PROVIDER_NAME = "Anthropic"
    ENV_API_KEY_NAME = "ANTHROPIC_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.anthropic.com/en/home"

    SUPPORTS_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_REASONING = False
    SUPPORTS_EMBEDDING = False

    @classmethod
    def verify_kwargs(cls, kwargs: dict[str, Any]) -> None:
        if kwargs.get("stream", False) and kwargs.get("response_format", None):
            raise UnsupportedParameterError("stream and response_format", cls.PROVIDER_NAME)

    def _stream_completion(
        self,
        client: Anthropic,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        # Convert messages for Anthropic format
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

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Anthropic with instructor support."""

        client = Anthropic(api_key=self.config.api_key, base_url=self.config.api_base)
        kwargs = _convert_kwargs(kwargs)

        if "response_format" in kwargs:
            if kwargs.get("stream", False):
                raise UnsupportedParameterError("response_format with streaming", self.PROVIDER_NAME)
            instructor_client = instructor.from_anthropic(client)

            response_format = kwargs.pop("response_format")

            # Convert messages for Anthropic format
            system_message, filtered_messages = _convert_messages_for_anthropic(messages)

            # Prepare kwargs for instructor
            instructor_kwargs = kwargs.copy()
            if system_message:
                instructor_kwargs["system"] = system_message

            # Use instructor for structured output
            instructor_response = instructor_client.messages.create(
                model=model,
                messages=filtered_messages,  # type: ignore[arg-type]
                response_model=response_format,
                **instructor_kwargs,
            )

            return convert_instructor_response(instructor_response, model, self.PROVIDER_NAME)

        if kwargs.get("stream", False):
            kwargs.pop("stream")
            return self._stream_completion(client, model, messages, **kwargs)
        else:
            # Convert messages for Anthropic format
            system_message, filtered_messages = _convert_messages_for_anthropic(messages)

            # Prepare kwargs for Anthropic
            anthropic_kwargs = kwargs.copy()
            if system_message:
                anthropic_kwargs["system"] = system_message

            message = client.messages.create(
                model=model,
                messages=filtered_messages,  # type: ignore[arg-type]
                **anthropic_kwargs,
            )
            return _convert_response(message)
