import re
from collections.abc import AsyncIterator, Callable, Collection
from typing import Any, Literal, TypeVar

from any_llm.constants import REASONING_FIELD_NAMES
from any_llm.utils.aio import aclose_quietly

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


def partial_reasoning_tag_suffix_len(text: str, *, tag_kind: Literal["opening", "closing"]) -> int:
    """Length of the longest suffix of text that is a proper prefix of a reasoning tag.

    A non-zero result means the tail of ``text`` may become a complete reasoning tag once
    more content arrives, so it has to stay buffered instead of being flushed downstream.
    Returns 0 when no suffix could grow into a tag.
    """
    longest = 0
    for tag_name in REASONING_FIELD_NAMES:
        tag = f"<{tag_name}>" if tag_kind == "opening" else f"</{tag_name}>"
        for i in range(min(len(text), len(tag) - 1), 0, -1):
            if text.endswith(tag[:i]):
                longest = max(longest, i)
                break
    return longest


async def process_streaming_reasoning_chunks(
    chunks: AsyncIterator[T],
    get_content: Callable[[T], str | None],
    set_content: Callable[[T, str | None], T],
    set_reasoning: Callable[[T, str], T],
    is_terminal: Callable[[T], bool],
    clear_stream_metadata: Callable[[T], T],
) -> AsyncIterator[T]:
    """Process streaming chunks to extract reasoning from XML tags.

    This async generator handles buffering across chunks to properly detect
    and extract reasoning tags that may be split across multiple chunks.

    Args:
        chunks: Async iterator of chunks to process
        get_content: Callable that extracts content string from a chunk (returns str | None)
        set_content: Callable that sets content on a chunk copy (chunk, content) -> chunk
        set_reasoning: Callable that sets reasoning on a chunk copy (chunk, reasoning) -> chunk
        is_terminal: Callable that reports whether a chunk carries a finish reason
        clear_stream_metadata: Callable that removes metadata from a synthetic flush chunk

    Yields:
        Processed chunks with reasoning extracted and separated from content

    """
    buffer = ""
    current_tag = None
    reasoning_buffer = ""
    last_chunk: T | None = None
    terminal_chunk: T | None = None
    terminal_content_parts: list[str] = []
    terminal_reasoning_parts: list[str] = []
    held_chunks: list[T] = []
    last_chunk_was_yielded = False

    try:
        async for original_chunk in chunks:
            content = get_content(original_chunk)

            if not content:
                if is_terminal(original_chunk) and (buffer or reasoning_buffer):
                    held_chunks.append(original_chunk)
                else:
                    yield original_chunk
                continue

            last_chunk = original_chunk
            last_chunk_was_yielded = False
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
                    else:
                        partial_len = partial_reasoning_tag_suffix_len(buffer, tag_kind="opening")
                        if partial_len:
                            if partial_len < len(buffer):
                                content_parts.append(buffer[:-partial_len])
                            buffer = buffer[len(buffer) - partial_len :]
                            break
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
                    else:
                        partial_len = partial_reasoning_tag_suffix_len(buffer, tag_kind="closing")
                        if partial_len:
                            reasoning_buffer += buffer[: len(buffer) - partial_len]
                            buffer = buffer[len(buffer) - partial_len :]
                            break
                        reasoning_buffer += buffer
                        buffer = ""

            if is_terminal(original_chunk) and (buffer or reasoning_buffer):
                terminal_chunk = original_chunk
                terminal_content_parts.extend(content_parts)
                terminal_reasoning_parts.extend(reasoning_parts)
                continue

            if content_parts or reasoning_parts:
                modified_chunk = original_chunk.model_copy(deep=True)  # type: ignore[attr-defined]
                modified_chunk = set_content(modified_chunk, "".join(content_parts) if content_parts else None)
                if reasoning_parts:
                    modified_chunk = set_reasoning(modified_chunk, "".join(reasoning_parts))
                yield modified_chunk
                last_chunk_was_yielded = True
            elif not buffer:
                modified_chunk = original_chunk.model_copy(deep=True)  # type: ignore[attr-defined]
                modified_chunk = set_content(modified_chunk, None)
                yield modified_chunk
                last_chunk_was_yielded = True
    finally:
        await aclose_quietly(chunks)

    if terminal_chunk is not None:
        final_chunk = terminal_chunk.model_copy(deep=True)  # type: ignore[attr-defined]
        final_content = "".join(terminal_content_parts)
        final_reasoning = "".join(terminal_reasoning_parts)
        if current_tag is None:
            final_content += buffer
        else:
            final_reasoning += reasoning_buffer + buffer
        final_chunk = set_content(
            final_chunk,
            final_content or None,
        )
        if final_reasoning:
            final_chunk = set_reasoning(final_chunk, final_reasoning)
        yield final_chunk
    elif last_chunk is not None and (buffer or reasoning_buffer):
        final_chunk = last_chunk.model_copy(deep=True)  # type: ignore[attr-defined]
        if last_chunk_was_yielded:
            final_chunk = clear_stream_metadata(final_chunk)
        if current_tag is None:
            final_chunk = set_content(final_chunk, buffer)
        else:
            final_chunk = set_content(final_chunk, None)
            final_chunk = set_reasoning(final_chunk, reasoning_buffer + buffer)
        yield final_chunk

    for held_chunk in held_chunks:
        yield held_chunk


def _without_extra_content(container: dict[str, Any], keep_namespaces: Collection[str]) -> dict[str, Any]:
    """Return ``container`` without the ``extra_content`` namespaces outside ``keep_namespaces``."""
    if "extra_content" not in container:
        return container
    extra_content = container["extra_content"]
    kept = (
        {namespace: value for namespace, value in extra_content.items() if namespace in keep_namespaces}
        if isinstance(extra_content, dict)
        else {}
    )
    if kept and kept == extra_content:
        return container
    cleaned = {key: value for key, value in container.items() if key != "extra_content"}
    if kept:
        cleaned["extra_content"] = kept
    return cleaned


def strip_extra_content(
    messages: list[dict[str, Any]], *, keep_namespaces: Collection[str] = ()
) -> list[dict[str, Any]]:
    """Drop the ``extra_content`` side-channel from messages and from their tool calls.

    any_llm keeps provider signatures and replayed reasoning in ``extra_content``: on a message
    (Anthropic thinking signatures, DeepSeek reasoning) and on a tool call (Gemini thought
    signatures). The OpenAI schema has no such field, the OpenAI SDK forwards unknown message keys
    verbatim, and strict OpenAI-compatible backends reject the whole request over one, so a
    conversation carrying another provider's signature would fail on its next turn.

    Namespaces in ``keep_namespaces`` stay, because for some backends they are the wire format:
    Gemini's OpenAI-compatible API reads a replayed tool call's thought signature from
    ``extra_content["google"]``. The key is dropped outright when nothing is kept, since an empty
    ``extra_content`` is still an unknown key. A provider that reads the side-channel has to do so
    before calling this. The input is never mutated, and a message or tool call with nothing to
    strip is returned as the same object.
    """
    result = []
    for message in messages:
        cleaned = _without_extra_content(message, keep_namespaces)
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            cleaned_calls = [
                _without_extra_content(call, keep_namespaces) if isinstance(call, dict) else call for call in tool_calls
            ]
            if any(new is not old for new, old in zip(cleaned_calls, tool_calls, strict=True)):
                cleaned = {**cleaned, "tool_calls": cleaned_calls}
        result.append(cleaned)
    return result


def replay_reasoning_content_as_reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rename a replayed ``reasoning_content`` to ``reasoning`` and drop ``extra_content``.

    For providers whose SDK names the assistant reasoning field ``reasoning`` and whose API rejects
    any other message key: Groq and Cerebras return 400 ``property 'reasoning_content' is
    unsupported``. The Messages bridge emits ``reasoning_content`` from a replayed ``thinking``
    block, and a caller of plain ``completion()`` may send it directly in the shape DeepSeek
    expects; both are renamed. An explicit ``reasoning`` the caller already set wins over
    ``reasoning_content``.
    """
    result = []
    for message in strip_extra_content(messages):
        if "reasoning_content" not in message:
            result.append(message)
            continue
        cleaned = {key: value for key, value in message.items() if key != "reasoning_content"}
        reasoning_content = message["reasoning_content"]
        if isinstance(reasoning_content, str) and reasoning_content and "reasoning" not in cleaned:
            cleaned["reasoning"] = reasoning_content
        result.append(cleaned)
    return result


def normalize_reasoning_from_provider_fields_and_xml_tags(message_dict: dict[str, Any]) -> None:
    """Extract and normalize reasoning from provider fields and XML tags.

    This function mutates the message_dict in place:
    1. First checks for reasoning in provider-specific fields (reasoning_content, thinking, etc.)
    2. Then extracts reasoning from XML tags in content in textual order (<think>, <thinking>, etc.)
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
        tag_names = "|".join(re.escape(tag_name) for tag_name in REASONING_FIELD_NAMES)
        think_pattern = re.compile(rf"<(?P<tag>{tag_names})>(?P<reasoning>.*?)</(?P=tag)>", re.DOTALL)
        matches = [match.group("reasoning") for match in think_pattern.finditer(content)]
        if matches:
            # Empty blocks must not become a nonempty reasoning string just from joining them.
            extracted_reasoning = "\n".join(matches) if any(matches) else ""
            if reasoning_content:
                reasoning_content = f"{reasoning_content}\n{extracted_reasoning}"
            else:
                reasoning_content = extracted_reasoning
            content = think_pattern.sub("", content).strip()

        message_dict["content"] = content

    if reasoning_content is not None:
        message_dict["reasoning"] = {"content": str(reasoning_content)}
