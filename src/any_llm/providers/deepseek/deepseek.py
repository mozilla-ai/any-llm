from collections.abc import AsyncIterator
from typing import Any

from typing_extensions import override

from any_llm.providers.deepseek.utils import _inject_cached_tokens, _inject_cached_tokens_chunk, _preprocess_messages
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class DeepseekProvider(BaseOpenAIProvider):
    API_BASE = "https://api.deepseek.com"
    ENV_API_KEY_NAME = "DEEPSEEK_API_KEY"
    ENV_API_BASE_NAME = "DEEPSEEK_API_BASE"
    PROVIDER_NAME = "deepseek"
    PROVIDER_DOCUMENTATION_URL = "https://platform.deepseek.com/"

    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False  # DeepSeek doesn't host an embedding model
    SUPPORTS_COMPLETION_REASONING = True

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        result = BaseOpenAIProvider._convert_completion_response(response)
        return _inject_cached_tokens(result)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        result = BaseOpenAIProvider._convert_completion_chunk_response(response, **kwargs)
        return _inject_cached_tokens_chunk(result)

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        return await super()._acompletion(_preprocess_messages(params), **kwargs)
