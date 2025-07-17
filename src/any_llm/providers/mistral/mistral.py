import os
from typing import Any, Iterator

try:
    from mistralai import Mistral
    from mistralai.extra import response_format_from_pydantic_model
except ImportError:
    msg = "mistralai is not installed. Please install it with `pip install any-llm-sdk[mistral]`"
    raise ImportError(msg)

from pydantic import BaseModel

from any_llm.types import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.helpers import create_completion_from_response
from any_llm.types import Stream
from any_llm.types import ChatCompletionChunk


class MistralProvider(Provider):
    """Mistral Provider using the new response conversion utilities."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Mistral provider."""
        if not config.api_key:
            config.api_key = os.getenv("MISTRAL_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError("Mistral", "MISTRAL_API_KEY")

        self.client = Mistral(api_key=config.api_key, server_url=config.api_base)

    def _stream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        # Get the Mistral stream
        mistral_stream = self.client.chat.stream(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        for event in mistral_stream:
            from any_llm.providers.mistral.utils import _create_openai_chunk_from_mistral_chunk

            yield _create_openai_chunk_from_mistral_chunk(event)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Mistral."""

        # Handle response_format for Pydantic models
        if "response_format" in kwargs and issubclass(kwargs["response_format"], BaseModel):
            kwargs["response_format"] = response_format_from_pydantic_model(kwargs["response_format"])

        if not kwargs.get("stream", False):
            response = self.client.chat.complete(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )

            # Convert to OpenAI format using the new utility
            return create_completion_from_response(
                response_data=response.model_dump(),
                model=model,
                provider_name="mistral",
            )
        else:
            # Return the streaming generator
            return self._stream_completion(model, messages, **kwargs)  # type: ignore[return-value]
