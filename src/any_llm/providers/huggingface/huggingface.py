from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from any_llm.provider import Provider
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionParams,
    CompletionUsage,
)
from any_llm.types.model import Model

MISSING_PACKAGES_ERROR = None
try:
    from huggingface_hub import AsyncInferenceClient, HfApi, InferenceClient

    from .utils import (
        _convert_models_list,
        _convert_params,
        _create_openai_chunk_from_huggingface_chunk,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from huggingface_hub.inference._generated.types import (  # type: ignore[attr-defined]
        ChatCompletionStreamOutput as HuggingFaceChatCompletionStreamOutput,
    )


class HuggingfaceProvider(Provider):
    """HuggingFace Provider using the new response conversion utilities."""

    PROVIDER_NAME = "huggingface"
    ENV_API_KEY_NAME = "HF_TOKEN"
    PROVIDER_DOCUMENTATION_URL = "https://huggingface.co/docs/huggingface_hub/package_reference/inference_client"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    async def _stream_completion_async(
        self,
        client: "AsyncInferenceClient",
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        response: AsyncIterator[HuggingFaceChatCompletionStreamOutput] = await client.chat_completion(**kwargs)

        async for chunk in response:
            yield _create_openai_chunk_from_huggingface_chunk(chunk)

    def _stream_completion(
        self,
        client: "InferenceClient",
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        response: Iterator[HuggingFaceChatCompletionStreamOutput] = client.chat_completion(
            **kwargs,
        )
        for chunk in response:
            yield _create_openai_chunk_from_huggingface_chunk(chunk)

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using HuggingFace."""
        client = AsyncInferenceClient(
            base_url=self.config.api_base, token=self.config.api_key, timeout=kwargs.get("timeout")
        )

        converted_kwargs = _convert_params(params, **kwargs)

        if params.stream:
            converted_kwargs["stream"] = True
            return self._stream_completion_async(client, **converted_kwargs)

        response = await client.chat_completion(**converted_kwargs)

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
            model=params.model_id,
            created=data.get("created", 0),
            object="chat.completion",
            choices=choices_out,
            usage=usage,
        )

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using HuggingFace."""
        client = InferenceClient(
            base_url=self.config.api_base, token=self.config.api_key, timeout=kwargs.get("timeout")
        )

        converted_kwargs = _convert_params(params, **kwargs)

        if params.stream:
            converted_kwargs["stream"] = True
            return self._stream_completion(client, **converted_kwargs)

        response = client.chat_completion(**converted_kwargs)

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
            model=params.model_id,
            created=data.get("created", 0),
            object="chat.completion",
            choices=choices_out,
            usage=usage,
        )

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        if not self.SUPPORTS_LIST_MODELS:
            message = f"{self.PROVIDER_NAME} does not support listing models."
            raise NotImplementedError(message)
        client = HfApi(token=self.config.api_key)
        if kwargs.get("inference") is None and kwargs.get("inference_provider") is None:
            kwargs["inference"] = "warm"
        if kwargs.get("limit") is None:
            kwargs["limit"] = 20
        models_list = client.list_models(**kwargs)
        return _convert_models_list(models_list)
