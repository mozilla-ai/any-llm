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

    SUPPORTS_STREAMING = True

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Default is that all kwargs are supported."""
        pass

    def _make_api_call(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Make the API call to OpenAI-compatible service."""
        # Create the OpenAI client
        client = OpenAI(
            base_url=self.config.api_base or self.API_BASE or os.getenv("OPENAI_API_BASE"),
            api_key=self.config.api_key,
        )

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
