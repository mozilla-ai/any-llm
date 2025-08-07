from typing import Any, Iterator

try:
    from huggingface_hub import InferenceClient
    from huggingface_hub.inference._generated.types import (  # type: ignore[attr-defined]
        ChatCompletionStreamOutput as HuggingFaceChatCompletionStreamOutput,
    )
except ImportError as exc:
    msg = "huggingface-hub is not installed. Please install it with `pip install any-llm-sdk[huggingface]`"
    raise ImportError(msg) from exc

from pydantic import BaseModel

from any_llm.types.completion import ChatCompletionChunk, ChatCompletion
from any_llm.provider import Provider
from any_llm.providers.helpers import (
    create_completion_from_response,
)
from any_llm.providers.huggingface.utils import (
    _convert_pydantic_to_huggingface_json,
    _create_openai_chunk_from_huggingface_chunk,
)


class HuggingfaceProvider(Provider):
    """HuggingFace Provider using the new response conversion utilities."""

    PROVIDER_NAME = "HuggingFace"
    ENV_API_KEY_NAME = "HF_TOKEN"
    PROVIDER_DOCUMENTATION_URL = "https://huggingface.co/inference-endpoints"

    SUPPORTS_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_REASONING = False
    SUPPORTS_EMBEDDING = False

    @classmethod
    def verify_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the HuggingFace provider."""

    def _stream_completion(
        self,
        client: InferenceClient,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        response: Iterator[HuggingFaceChatCompletionStreamOutput] = client.chat_completion(
            model=model,
            messages=messages,
            **kwargs,
        )
        for chunk in response:
            yield _create_openai_chunk_from_huggingface_chunk(chunk)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using HuggingFace."""
        client = InferenceClient(token=self.config.api_key, timeout=kwargs.get("timeout", None))

        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")

        # Handle response_format for Pydantic models
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                messages = _convert_pydantic_to_huggingface_json(response_format, messages)

        # Handle streaming
        if kwargs.get("stream", False):
            return self._stream_completion(client, model, messages, **kwargs)

        # Make the non-streaming API call
        response = client.chat_completion(
            model=model,
            messages=messages,
            **kwargs,
        )

        return create_completion_from_response(
            response_data=response,
            model=model,
            provider_name=self.PROVIDER_NAME,
        )
