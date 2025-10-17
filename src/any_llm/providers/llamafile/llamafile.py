from collections.abc import AsyncIterator
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.llamafile.utils import _convert_chat_completion
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class LlamafileProvider(BaseOpenAIProvider):
    API_BASE = "http://127.0.0.1:8080/v1"
    ENV_API_KEY_NAME = "None"
    PROVIDER_NAME = "llamafile"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/Mozilla-Ocho/llamafile"

    SUPPORTS_EMBEDDING = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_STREAMING = False
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False

    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        return ""

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        # Overriding the base behavior so that we can use our custom parsing of reasoning content
        if isinstance(response, OpenAIChatCompletion):
            return _convert_chat_completion(response)
        if isinstance(response, ChatCompletion):
            return response
        return ChatCompletion.model_validate(response)

    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
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
        return await super()._acompletion(params, **kwargs)
