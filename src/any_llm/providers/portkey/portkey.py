from collections.abc import AsyncIterator
from typing import Any

from openai._streaming import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from pydantic import BaseModel

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.portkey.utils import _convert_chat_completion, _convert_chat_completion_chunk
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, Reasoning
from any_llm.utils.reasoning import (
    process_streaming_reasoning_chunks,
)


class PortkeyProvider(BaseOpenAIProvider):
    """Portkey provider for accessing 200+ LLMs through Portkey's AI Gateway."""

    API_BASE = "https://api.portkey.ai/v1"
    ENV_API_KEY_NAME = "PORTKEY_API_KEY"
    PROVIDER_NAME = "portkey"
    PROVIDER_DOCUMENTATION_URL = "https://portkey.ai/docs"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    _DEFAULT_REASONING_EFFORT = None

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        if isinstance(response, OpenAIChatCompletion):
            return _convert_chat_completion(response)
        if isinstance(response, ChatCompletion):
            return response
        return ChatCompletion.model_validate(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        if isinstance(response, OpenAIChatCompletionChunk):
            return _convert_chat_completion_chunk(response)
        if isinstance(response, ChatCompletionChunk):
            return response
        return ChatCompletionChunk.model_validate(response)

    def _convert_completion_response_async(
        self, response: OpenAIChatCompletion | AsyncStream[OpenAIChatCompletionChunk]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Convert an OpenAI completion response with streaming reasoning support."""
        if isinstance(response, OpenAIChatCompletion):
            return self._convert_completion_response(response)

        async def chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in response:
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
        """Convert CompletionParams to kwargs for OpenAI API."""
        if isinstance(params.response_format, type) and issubclass(params.response_format, BaseModel):
            params.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": params.response_format.model_json_schema(),
                },
            }
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages"})
        converted_params.update(kwargs)
        return converted_params
