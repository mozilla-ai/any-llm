from collections.abc import AsyncIterator
from typing import Any

from any_llm.providers.deepseek.utils import _preprocess_messages
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class DeepseekProvider(BaseOpenAIProvider):
    API_BASE = "https://api.deepseek.com"
    ENV_API_KEY_NAME = "DEEPSEEK_API_KEY"
    PROVIDER_NAME = "deepseek"
    PROVIDER_DOCUMENTATION_URL = "https://platform.deepseek.com/"

    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False  # DeepSeek doesn't host an embedding model
    SUPPORTS_COMPLETION_REASONING = True

    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        return await super()._acompletion(_preprocess_messages(params), **kwargs)
