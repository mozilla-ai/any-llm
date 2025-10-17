import re
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion

from any_llm.constants import REASONING_FIELD_NAMES
from any_llm.providers.openai.utils import _normalize_openai_dict_response
from any_llm.types.completion import ChatCompletion


def _extract_reasoning_from_xml_tags(message_dict: dict[str, Any]) -> None:
    """Extract reasoning content from XML tags in the message content field."""
    content = message_dict.get("content")
    if not isinstance(content, str):
        return

    reasoning_dict = message_dict.get("reasoning")
    existing_reasoning = reasoning_dict.get("content") if isinstance(reasoning_dict, dict) else None

    for tag_name in REASONING_FIELD_NAMES:
        tag_open = f"<{tag_name}>"
        tag_close = f"</{tag_name}>"
        think_pattern = re.escape(tag_open) + r"(.*?)" + re.escape(tag_close)
        matches = re.findall(think_pattern, content, re.DOTALL)
        if matches:
            extracted_reasoning = "\n".join(matches)
            if existing_reasoning:
                existing_reasoning = f"{existing_reasoning}\n{extracted_reasoning}"
            else:
                existing_reasoning = extracted_reasoning
            content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()

    message_dict["content"] = content
    if existing_reasoning is not None:
        message_dict["reasoning"] = {"content": existing_reasoning}


def _convert_chat_completion(response: OpenAIChatCompletion) -> ChatCompletion:
    response_dict = _normalize_openai_dict_response(response.model_dump())

    choices = response_dict.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                _extract_reasoning_from_xml_tags(message)

            delta = choice.get("delta") if isinstance(choice, dict) else None
            if isinstance(delta, dict):
                _extract_reasoning_from_xml_tags(delta)

    return ChatCompletion.model_validate(response_dict)
