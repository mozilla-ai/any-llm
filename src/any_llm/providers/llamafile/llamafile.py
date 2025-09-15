import os
from collections.abc import AsyncIterator
from typing import Any

from any_llm.config import ClientConfig
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class LlamafileProvider(BaseOpenAIProvider):
    API_BASE = "http://127.0.0.1:8080/v1"
    ENV_API_KEY_NAME = "None"
    PROVIDER_NAME = "llamafile"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/Mozilla-Ocho/llamafile"

    SUPPORTS_EMBEDDING = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_STREAMING = False
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False

    def __init__(self, config: ClientConfig) -> None:
        """We don't use the Provider init because by default we don't require an API key."""

        self.url = config.api_base or os.getenv("LLAMAFILE_API_URL")
        self.config = config
        self.config.api_key = ""  # In order to be compatible with the OpenAI client, the API key cannot be None if the OPENAI_API_KEY environment variable is not set (which is the case for LlamaFile)

    async def acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Handle completion - extracted to avoid generator issues."""
        if params.response_format:
            msg = "response_format"
            raise UnsupportedParameterError(
                msg,
                self.PROVIDER_NAME,
            )
        if params.tools:
            msg = "tools"
            raise UnsupportedParameterError(
                msg,
                self.PROVIDER_NAME,
            )
        return await super().acompletion(params, **kwargs)
