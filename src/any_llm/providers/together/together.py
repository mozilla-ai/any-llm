from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from any_llm.any_llm import AnyLLM

MISSING_PACKAGES_ERROR = None
try:
    import together

    from .utils import (
        _convert_together_response_to_chat_completion,
        _create_openai_chunk_from_together_chunk,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from together.types import (
        ChatCompletionResponse,
    )
    from together.types.chat_completions import ChatCompletionChunk as TogetherChatCompletionChunk

    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionChunk,
        CompletionParams,
        CreateEmbeddingResponse,
    )
    from any_llm.types.model import Model


class TogetherProvider(AnyLLM):
    PROVIDER_NAME = "together"
    ENV_API_KEY_NAME = "TOGETHER_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://together.ai/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: together.AsyncTogether

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Together API."""
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format"})
        if converted_params.get("reasoning_effort") == "auto":
            converted_params.pop("reasoning_effort")
        if (
            params.response_format is not None
            and isinstance(params.response_format, type)
            and issubclass(params.response_format, BaseModel)
        ):
            converted_params["response_format"] = {
                "type": "json_schema",
                "schema": params.response_format.model_json_schema(),
            }

        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Together response to OpenAI format."""
        # We need the model parameter for conversion
        model = response.get("model", "together-model")
        return _convert_together_response_to_chat_completion(response, model)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Together chunk response to OpenAI format."""
        return _create_openai_chunk_from_together_chunk(response)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Together."""
        msg = "Together does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Together embedding response to OpenAI format."""
        msg = "Together does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Together list models response to OpenAI format."""
        msg = "Together does not support listing models"
        raise NotImplementedError(msg)

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = together.AsyncTogether(
            api_key=api_key,
            base_url=api_base,
            **kwargs,
        )

    async def _stream_completion_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion with reasoning support."""
        from typing import cast

        response = cast(
            "AsyncIterator[TogetherChatCompletionChunk]",
            await self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            ),
        )

        async for chunk in response:
            yield self._convert_completion_chunk_response(chunk)

    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return self._stream_completion_async(
                params.model_id,
                params.messages,
                **completion_kwargs,
            )

        response = cast(
            "ChatCompletionResponse",
            await self.client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **completion_kwargs,
            ),
        )

        return self._convert_completion_response(response.model_dump())
