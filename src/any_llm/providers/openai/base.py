import os
from typing import Any
from abc import ABC

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from any_llm.provider import Provider


class BaseOpenAIProvider(Provider, ABC):
    """
    Base provider for OpenAI-compatible services.

    This class provides a common foundation for providers that use OpenAI-compatible APIs.
    Subclasses only need to override configuration defaults and client initialization
    if needed.
    """

    # Extended api_base which isn't required by the Provider class
    DEFAULT_API_BASE: str | None = None

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Default is that all kwargs are supported."""
        pass

    def _make_api_call(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Make the API call to OpenAI-compatible service."""
        client_kwargs: dict[str, Any] = {}

        if not self.config.api_base:
            client_kwargs["base_url"] = self.DEFAULT_API_BASE or os.getenv("OPENAI_API_BASE")
        else:
            client_kwargs["base_url"] = self.config.api_base

        # API key is already validated in Provider
        client_kwargs["api_key"] = self.config.api_key

        # Create the OpenAI client
        client = OpenAI(**client_kwargs)

        if "response_format" in kwargs:
            response = client.chat.completions.parse(  # type: ignore[attr-defined]
                model=model,
                messages=messages,
                **kwargs,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )
        return response  # type: ignore[no-any-return]
