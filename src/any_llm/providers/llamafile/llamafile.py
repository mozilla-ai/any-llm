from collections.abc import AsyncIterator
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from typing_extensions import override

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.openai.xml_reasoning_utils import convert_chat_completion_with_xml_reasoning
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class LlamafileProvider(BaseOpenAIProvider):
    API_BASE = "http://127.0.0.1:8080/v1"
    ENV_API_KEY_NAME = "None"
    ENV_API_BASE_NAME = "LLAMAFILE_API_BASE"
    PROVIDER_NAME = "llamafile"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/Mozilla-Ocho/llamafile"

    SUPPORTS_EMBEDDING = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_STREAMING = False
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        return ""

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        if isinstance(response, OpenAIChatCompletion):
            return convert_chat_completion_with_xml_reasoning(response)
        if isinstance(response, ChatCompletion):
            return response
        return ChatCompletion.model_validate(response)

    @override
    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if params.response_format:
            msg = "response_format"
            raise UnsupportedParameterError(
                msg,
                self.PROVIDER_NAME,
            )
        return await super()._acompletion(params, **kwargs)
