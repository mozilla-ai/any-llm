from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

try:
    from huggingface_hub import InferenceClient

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from pydantic import BaseModel

from any_llm.provider import Provider
from any_llm.providers.huggingface.utils import (
    _convert_pydantic_to_huggingface_json,
    _create_openai_chunk_from_huggingface_chunk,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage, Choice, CompletionUsage

if TYPE_CHECKING:
    from huggingface_hub.inference._generated.types import (  # type: ignore[attr-defined]
        ChatCompletionStreamOutput as HuggingFaceChatCompletionStreamOutput,
    )


class HuggingfaceProvider(Provider):
    """HuggingFace Provider using the new response conversion utilities."""

    PROVIDER_NAME = "huggingface"
    ENV_API_KEY_NAME = "HF_TOKEN"
    PROVIDER_DOCUMENTATION_URL = "https://huggingface.co/inference-endpoints"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    def _stream_completion(
        self,
        client: "InferenceClient",
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

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using HuggingFace."""
        client = InferenceClient(token=self.config.api_key, timeout=kwargs.get("timeout", None))

        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")

        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                messages = _convert_pydantic_to_huggingface_json(response_format, messages)

        if kwargs.get("stream", False):
            return self._stream_completion(client, model, messages, **kwargs)

        response = client.chat_completion(
            model=model,
            messages=messages,
            **kwargs,
        )
        data = response
        choices_out: list[Choice] = []
        for i, ch in enumerate(data.get("choices", [])):
            msg = ch.get("message", {})
            message = ChatCompletionMessage(
                role="assistant",
                content=msg.get("content"),
                tool_calls=msg.get("tool_calls"),
            )
            choices_out.append(Choice(index=i, finish_reason=ch.get("finish_reason"), message=message))

        usage = None
        if data.get("usage"):
            u = data["usage"]
            usage = CompletionUsage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            )

        return ChatCompletion(
            id=data.get("id", ""),
            model=model,
            created=data.get("created", 0),
            object="chat.completion",
            choices=choices_out,
            usage=usage,
        )
