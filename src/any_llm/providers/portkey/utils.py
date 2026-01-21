from typing import TYPE_CHECKING

from any_llm.providers.openai.utils import _normalize_openai_dict_response
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.utils.reasoning import normalize_reasoning_from_provider_fields_and_xml_tags

if TYPE_CHECKING:
    from portkey_ai.api_resources.types.chat_complete_type import (
        ChatCompletionChunk as PortkeyChatCompletionChunk,
    )
    from portkey_ai.api_resources.types.chat_complete_type import (
        ChatCompletions as PortkeyChatCompletions,
    )


def _convert_chat_completion(response: "PortkeyChatCompletions") -> ChatCompletion:
    response_dict = _normalize_openai_dict_response(response.model_dump())

    choices = response_dict.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                normalize_reasoning_from_provider_fields_and_xml_tags(message)

    return ChatCompletion.model_validate(response_dict)


def _convert_chat_completion_chunk(response: "PortkeyChatCompletionChunk") -> ChatCompletionChunk:
    response_dict = _normalize_openai_dict_response(response.model_dump())

    choices = response_dict.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            delta = choice.get("delta") if isinstance(choice, dict) else None
            if isinstance(delta, dict):
                normalize_reasoning_from_provider_fields_and_xml_tags(delta)

    response_dict["object"] = "chat.completion.chunk"
    return ChatCompletionChunk.model_validate(response_dict)
