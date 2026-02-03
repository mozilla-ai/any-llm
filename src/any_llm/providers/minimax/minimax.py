from collections.abc import AsyncIterator
from typing import Any

from openai._streaming import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from typing_extensions import override

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.openai.xml_reasoning import XMLReasoningOpenAIProvider, wrap_chunks_with_xml_reasoning
from any_llm.providers.openai.xml_reasoning_utils import convert_chat_completion_with_xml_reasoning
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class MinimaxProvider(XMLReasoningOpenAIProvider):
    API_BASE = "https://api.minimax.io/v1"
    ENV_API_KEY_NAME = "MINIMAX_API_KEY"
    ENV_API_BASE_NAME = "MINIMAX_API_BASE"
    PROVIDER_NAME = "minimax"
    PROVIDER_DOCUMENTATION_URL = "https://www.minimax.io/platform_overview"

    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_EMBEDDING = False

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        if isinstance(response, OpenAIChatCompletion):
            return convert_chat_completion_with_xml_reasoning(response)
        if isinstance(response, ChatCompletion):
            return response
        return ChatCompletion.model_validate(response)

    @override
    def _convert_completion_response_async(
        self, response: OpenAIChatCompletion | AsyncStream[OpenAIChatCompletionChunk]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Convert completion response with Minimax-specific chunk filtering."""
        if isinstance(response, OpenAIChatCompletion):
            return self._convert_completion_response(response)

        async def chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in response:
                if isinstance(chunk, OpenAIChatCompletionChunk):
                    if chunk.choices and chunk.choices[0].delta:
                        yield self._convert_completion_chunk_response(chunk)

        return wrap_chunks_with_xml_reasoning(chunk_iterator())

    @staticmethod
    @override
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
