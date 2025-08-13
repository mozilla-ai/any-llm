from collections.abc import Iterator
from typing import Any

try:
    import instructor
    from anthropic import Anthropic

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import Provider
from any_llm.providers.anthropic.utils import (
    _convert_kwargs,
    _convert_messages_for_anthropic,
    _convert_response,
    _create_openai_chunk_from_anthropic_chunk,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
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

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Anthropic with instructor support."""

        client = Anthropic(api_key=self.config.api_key, base_url=self.config.api_base)
        kwargs = _convert_kwargs(kwargs)

        if "response_format" in kwargs:
            instructor_client = instructor.from_anthropic(client)

            response_format = kwargs.pop("response_format")

            system_message, filtered_messages = _convert_messages_for_anthropic(messages)

            instructor_kwargs = kwargs.copy()
            if system_message:
                instructor_kwargs["system"] = system_message

            instructor_response = instructor_client.messages.create(
                model=model,
                messages=filtered_messages,  # type: ignore[arg-type]
                response_model=response_format,
                **instructor_kwargs,
            )

            return _convert_instructor_response(instructor_response, model, self.PROVIDER_NAME)

        if kwargs.get("stream", False):
            kwargs.pop("stream")
            return self._stream_completion(client, model, messages, **kwargs)
        system_message, filtered_messages = _convert_messages_for_anthropic(messages)

        anthropic_kwargs = kwargs.copy()
        if system_message:
            anthropic_kwargs["system"] = system_message

        message = client.messages.create(
            model=model,
            messages=filtered_messages,  # type: ignore[arg-type]
            **anthropic_kwargs,
        )
        return _convert_response(message)
