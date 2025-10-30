from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion

from any_llm.providers.openai.utils import _normalize_openai_dict_response
from any_llm.types.completion import ChatCompletion
from any_llm.utils.reasoning import normalize_reasoning_from_provider_fields_and_xml_tags


def _convert_chat_completion(response: OpenAIChatCompletion) -> ChatCompletion:
    response_dict = _normalize_openai_dict_response(response.model_dump())

    choices = response_dict.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                normalize_reasoning_from_provider_fields_and_xml_tags(message)

            delta = choice.get("delta") if isinstance(choice, dict) else None
            if isinstance(delta, dict):
                normalize_reasoning_from_provider_fields_and_xml_tags(delta)

    return ChatCompletion.model_validate(response_dict)
