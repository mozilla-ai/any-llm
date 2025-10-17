import re
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from any_llm.constants import REASONING_FIELD_NAMES

T = TypeVar("T")


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


async def process_streaming_reasoning_chunks(
    chunks: AsyncIterator[T],
    get_content: Any,
    set_content: Any,
    set_reasoning: Any,
) -> AsyncIterator[T]:
    """Process streaming chunks to extract reasoning from XML tags.

    This async generator handles buffering across chunks to properly detect
    and extract reasoning tags that may be split across multiple chunks.

    Args:
        chunks: Async iterator of chunks to process
        get_content: Callable that extracts content string from a chunk (returns str | None)
        set_content: Callable that sets content on a chunk copy (chunk, content) -> chunk
        set_reasoning: Callable that sets reasoning on a chunk copy (chunk, reasoning) -> chunk

    Yields:
        Processed chunks with reasoning extracted and separated from content

    """
    buffer = ""
    current_tag = None
    reasoning_buffer = ""

    async for original_chunk in chunks:
        content = get_content(original_chunk)

        if not content:
            yield original_chunk
            continue

        buffer += content
        content_parts = []
        reasoning_parts = []

        while buffer:
            if current_tag is None:
                tag_info = find_reasoning_tag(buffer, opening=True)
                if tag_info:
                    tag_start, tag_name = tag_info
                    if tag_start > 0:
                        content_parts.append(buffer[:tag_start])
                    tag_full = f"<{tag_name}>"
                    buffer = buffer[tag_start + len(tag_full) :]
                    current_tag = tag_name
                elif is_partial_reasoning_tag(buffer, opening=True):
                    break
                else:
                    content_parts.append(buffer)
                    buffer = ""
            else:
                tag_close = f"</{current_tag}>"
                tag_end = buffer.find(tag_close)
                if tag_end != -1:
                    reasoning_parts.append(reasoning_buffer + buffer[:tag_end])
                    reasoning_buffer = ""
                    buffer = buffer[tag_end + len(tag_close) :]
                    current_tag = None
                elif is_partial_reasoning_tag(buffer, opening=False):
                    reasoning_buffer += buffer
                    buffer = ""
                    break
                else:
                    reasoning_buffer += buffer
                    buffer = ""

        if content_parts or reasoning_parts:
            modified_chunk = original_chunk.model_copy(deep=True)  # type: ignore[attr-defined]
            modified_chunk = set_content(modified_chunk, "".join(content_parts) if content_parts else None)
            if reasoning_parts:
                modified_chunk = set_reasoning(modified_chunk, "".join(reasoning_parts))
            yield modified_chunk
        elif not buffer:
            modified_chunk = original_chunk.model_copy(deep=True)  # type: ignore[attr-defined]
            modified_chunk = set_content(modified_chunk, None)
            yield modified_chunk


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
