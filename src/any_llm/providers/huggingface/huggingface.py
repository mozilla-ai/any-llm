import os
from typing import Any

try:
    from huggingface_hub import InferenceClient
except ImportError:
    msg = "huggingface-hub is not installed. Please install it with `pip install any-llm-sdk[huggingface]`"
    raise ImportError(msg)

from pydantic import BaseModel

from any_llm.types import Stream
from any_llm.types import ChatCompletionChunk
from any_llm.types import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.helpers import (
    create_completion_from_response,
)
from any_llm.providers.huggingface.utils import _convert_pydantic_to_huggingface_json


class HuggingfaceProvider(Provider):
    """HuggingFace Provider using the new response conversion utilities."""

    PROVIDER_NAME = "HuggingFace"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize HuggingFace provider."""
        if not config.api_key:
            config.api_key = os.getenv("HF_TOKEN")
        if not config.api_key:
            raise MissingApiKeyError("HuggingFace", "HF_TOKEN")

        self.client = InferenceClient(token=config.api_key, timeout=30)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using HuggingFace."""

        if kwargs.get("stream", False) is True:
            raise UnsupportedParameterError("stream", "HuggingFace")

        # Convert max_tokens to max_new_tokens (HuggingFace specific)
        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")

        # Handle response_format for Pydantic models
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Convert Pydantic model to HuggingFace JSON format
                messages = _convert_pydantic_to_huggingface_json(response_format, messages)

        # Make the API call
        response = self.client.chat_completion(
            model=model,
            messages=messages,
            **kwargs,
        )

        # Convert to OpenAI format using the new utility
        return create_completion_from_response(
            response_data=response,
            model=model,
            provider_name="huggingface",
        )
