import re
from typing import Any

from any_llm.constants import REASONING_FIELD_NAMES


def find_reasoning_tag(text: str, opening: bool = True) -> tuple[int, str] | None:
    """Find the first reasoning tag (opening or closing) in text.

    Returns (position, tag_name) or None if no tag found.
    """
    earliest_pos = len(text)
    earliest_tag = None

    for tag_name in REASONING_FIELD_NAMES:
        tag = f"<{tag_name}>" if opening else f"</{tag_name}>"
        pos = text.find(tag)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            earliest_tag = tag_name

    return (earliest_pos, earliest_tag) if earliest_tag else None


def is_partial_reasoning_tag(text: str, opening: bool = True) -> bool:
    """Check if text could be the start of any reasoning tag."""
    for tag_name in REASONING_FIELD_NAMES:
        tag = f"<{tag_name}>" if opening else f"</{tag_name}>"
        for i in range(1, len(tag) + 1):
            if text.startswith(tag[:i]):
                return True
    return False


def normalize_reasoning_from_provider_fields_and_xml_tags(message_dict: dict[str, Any]) -> None:
    """Extract and normalize reasoning from provider fields and XML tags.

    This function mutates the message_dict in place:
    1. First checks for reasoning in provider-specific fields (reasoning_content, thinking, etc.)
    2. Then extracts reasoning from XML tags in content (<think>, <thinking>, etc.)
    3. Combines both sources if both exist
    4. Removes XML tags from content and stores reasoning separately

    Args:
        message_dict: A dictionary representing a message with 'content' and
                     optionally 'reasoning' fields.

    """
    if isinstance(message_dict.get("reasoning"), dict) and "content" in message_dict["reasoning"]:
        return

    reasoning_content = None

    for field_name in REASONING_FIELD_NAMES:
        if field_name in message_dict and message_dict[field_name] is not None:
            reasoning_content = message_dict[field_name]
            break

    if reasoning_content is None and isinstance(message_dict.get("reasoning"), str):
        reasoning_content = message_dict["reasoning"]

    content = message_dict.get("content")
    if isinstance(content, str):
        for tag_name in REASONING_FIELD_NAMES:
            tag_open = f"<{tag_name}>"
            tag_close = f"</{tag_name}>"
            think_pattern = re.escape(tag_open) + r"(.*?)" + re.escape(tag_close)
            matches = re.findall(think_pattern, content, re.DOTALL)
            if matches:
                extracted_reasoning = "\n".join(matches)
                if reasoning_content:
                    reasoning_content = f"{reasoning_content}\n{extracted_reasoning}"
                else:
                    reasoning_content = extracted_reasoning
                content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()

        message_dict["content"] = content

    if reasoning_content is not None:
        message_dict["reasoning"] = {"content": str(reasoning_content)}
