from collections.abc import Iterator, AsyncIterator
from typing import Any

from pydantic import BaseModel

from any_llm.providers.deepseek.utils import _convert_pydantic_to_deepseek_json
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class DeepseekProvider(BaseOpenAIProvider):
    API_BASE = "https://api.deepseek.com"
    ENV_API_KEY_NAME = "DEEPSEEK_API_KEY"
    PROVIDER_NAME = "deepseek"
    PROVIDER_DOCUMENTATION_URL = "https://platform.deepseek.com/"

    SUPPORTS_EMBEDDING = False  # DeepSeek doesn't host an embedding model

    def _preprocess_messages(self, params: CompletionParams) -> CompletionParams:
        """Preprocess messages"""
        if params.response_format:
            if isinstance(params.response_format, type) and issubclass(params.response_format, BaseModel):
                modified_messages = _convert_pydantic_to_deepseek_json(params.response_format, params.messages)
                params.response_format = {"type": "json_object"}
                params.messages = modified_messages
        
        return params

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        return await super().acompletion(self._preprocess_messages(params), **kwargs)

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        return super().completion(self._preprocess_messages(params), **kwargs)
