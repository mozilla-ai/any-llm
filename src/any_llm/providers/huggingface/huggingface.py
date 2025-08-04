from typing import Any

try:
    from huggingface_hub import InferenceClient
except ImportError:
    msg = "huggingface-hub is not installed. Please install it with `pip install any-llm-sdk[huggingface]`"
    raise ImportError(msg)

from pydantic import BaseModel

from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.helpers import (
    create_completion_from_response,
)
from any_llm.providers.huggingface.utils import _convert_pydantic_to_huggingface_json


class HuggingfaceProvider(Provider):
    """HuggingFace Provider using the new response conversion utilities."""

    PROVIDER_NAME = "HuggingFace"
    ENV_API_KEY_NAME = "HF_TOKEN"
    PROVIDER_DOCUMENTATION_URL = "https://huggingface.co/inference-endpoints"

    SUPPORTS_STREAMING = False
    SUPPORTS_EMBEDDING = False

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the HuggingFace provider."""
        if kwargs.get("stream", False):
            raise UnsupportedParameterError("stream", self.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using HuggingFace."""
        client = InferenceClient(token=self.config.api_key, timeout=kwargs.get("timeout", None))

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
        response = client.chat_completion(
            model=model,
            messages=messages,
            **kwargs,
        )

        # Convert to OpenAI format using the new utility
        return create_completion_from_response(
            response_data=response,
            model=model,
            provider_name=self.PROVIDER_NAME,
        )
