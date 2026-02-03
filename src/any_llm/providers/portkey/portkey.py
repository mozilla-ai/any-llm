from typing import Any

from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from pydantic import BaseModel
from typing_extensions import override

from any_llm.providers.openai.xml_reasoning import XMLReasoningOpenAIProvider
from any_llm.providers.openai.xml_reasoning_utils import (
    convert_chat_completion_chunk_with_xml_reasoning,
    convert_chat_completion_with_xml_reasoning,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class PortkeyProvider(XMLReasoningOpenAIProvider):
    """Portkey provider for accessing 200+ LLMs through Portkey's AI Gateway."""

    API_BASE = "https://api.portkey.ai/v1"
    ENV_API_KEY_NAME = "PORTKEY_API_KEY"
    ENV_API_BASE_NAME = "PORTKEY_API_BASE"
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
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        if isinstance(response, OpenAIChatCompletion):
            return convert_chat_completion_with_xml_reasoning(response)
        if isinstance(response, ChatCompletion):
            return response
        return ChatCompletion.model_validate(response)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        if isinstance(response, OpenAIChatCompletionChunk):
            return convert_chat_completion_chunk_with_xml_reasoning(response)
        if isinstance(response, ChatCompletionChunk):
            return response
        return ChatCompletionChunk.model_validate(response)

    @staticmethod
    @override
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
