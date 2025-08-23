from collections.abc import AsyncIterator
from typing import Any

from any_llm.providers.llama.utils import _patch_json_schema
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class LlamaProvider(BaseOpenAIProvider):
    """Llama provider for accessing multiple LLMs through Llama's API."""

    API_BASE = "https://api.llama.com/compat/v1/"
    ENV_API_KEY_NAME = "LLAMA_API_KEY"
    PROVIDER_NAME = "llama"
    PROVIDER_DOCUMENTATION_URL = "https://www.llama.com/products/llama-api/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False

    async def acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if params.tools:
            params.tools = [_patch_json_schema(tool) for tool in params.tools]
        return await super().acompletion(params, **kwargs)
