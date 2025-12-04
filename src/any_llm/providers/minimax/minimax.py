from collections.abc import AsyncIterator
from typing import Any

from openai._streaming import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.minimax.utils import _convert_chat_completion
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, Reasoning
from any_llm.utils.reasoning import process_streaming_reasoning_chunks


class MinimaxProvider(BaseOpenAIProvider):
    API_BASE = "https://api.minimax.io/v1"
    ENV_API_KEY_NAME = "MINIMAX_API_KEY"
    PROVIDER_NAME = "minimax"
    PROVIDER_DOCUMENTATION_URL = "https://www.minimax.io/platform_overview"

    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_EMBEDDING = False

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        if isinstance(response, OpenAIChatCompletion):
            return _convert_chat_completion(response)
        if isinstance(response, ChatCompletion):
            return response
        return ChatCompletion.model_validate(response)

    def _convert_completion_response_async(
        self, response: OpenAIChatCompletion | AsyncStream[OpenAIChatCompletionChunk]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Convert an OpenAI completion response with streaming reasoning support."""
        if isinstance(response, OpenAIChatCompletion):
            return self._convert_completion_response(response)

        async def chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in response:
                if isinstance(chunk, OpenAIChatCompletionChunk):
                    if chunk.choices and chunk.choices[0].delta:
                        yield self._convert_completion_chunk_response(chunk)

        def get_content(chunk: ChatCompletionChunk) -> str | None:
            return chunk.choices[0].delta.content if len(chunk.choices) > 0 else None

        def set_content(chunk: ChatCompletionChunk, content: str | None) -> ChatCompletionChunk:
            chunk.choices[0].delta.content = content
            return chunk

        def set_reasoning(chunk: ChatCompletionChunk, reasoning: str) -> ChatCompletionChunk:
            chunk.choices[0].delta.reasoning = Reasoning(content=reasoning)
            return chunk

        return process_streaming_reasoning_chunks(
            chunk_iterator(),
            get_content=get_content,
            set_content=set_content,
            set_reasoning=set_reasoning,
        )

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        # response_format is supported in the z.ai SDK, but the SDK doesn't yet have an async client
        # so we can't use it in any-llm
        if params.response_format is not None:
            param = "response_format"
            raise UnsupportedParameterError(param, "minimax")
        # Copy of the logic from the base implementation because you can't use super() in a static method
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages"})
        converted_params.update(kwargs)
        return converted_params
