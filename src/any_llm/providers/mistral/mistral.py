from typing import Any, Iterator

try:
    from mistralai import Mistral
    from mistralai.extra import response_format_from_pydantic_model
except ImportError:
    msg = "mistralai is not installed. Please install it with `pip install any-llm-sdk[mistral]`"
    raise ImportError(msg)

from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider
from any_llm.providers.helpers import create_completion_from_response
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from any_llm.exceptions import UnsupportedParameterError


class MistralProvider(Provider):
    """Mistral Provider using the new response conversion utilities."""

    PROVIDER_NAME = "Mistral"
    ENV_API_KEY_NAME = "MISTRAL_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.mistral.ai/"

    def _stream_completion(
        self,
        client: Mistral,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        # Get the Mistral stream
        mistral_stream = client.chat.stream(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        for event in mistral_stream:
            from any_llm.providers.mistral.utils import _create_openai_chunk_from_mistral_chunk

            yield _create_openai_chunk_from_mistral_chunk(event)

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Mistral provider."""
        if kwargs.get("stream", False) is True:
            raise UnsupportedParameterError("stream", self.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Mistral."""
        client = Mistral(api_key=self.config.api_key, server_url=self.config.api_base)

        # Handle response_format for Pydantic models
        if "response_format" in kwargs and issubclass(kwargs["response_format"], BaseModel):
            kwargs["response_format"] = response_format_from_pydantic_model(kwargs["response_format"])

        if not kwargs.get("stream", False):
            response = client.chat.complete(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )

            # Convert to OpenAI format using the new utility
            return create_completion_from_response(
                response_data=response.model_dump(),
                model=model,
                provider_name=self.PROVIDER_NAME,
            )
        else:
            # Return the streaming generator
            return self._stream_completion(client, model, messages, **kwargs)  # type: ignore[return-value]
