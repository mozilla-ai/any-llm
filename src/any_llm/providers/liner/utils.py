from __future__ import annotations

import json
import time
from typing import Any, Literal

from any_llm.exceptions import ProviderError, UnsupportedParameterError
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
)

PROVIDER_NAME = "liner"

# Liner's agents fail with a bare 502 on any other role, so they are refused before the call.
_ROLES = frozenset({"user", "assistant"})

# Reference fields kept on a citation, under the names Exa's answer endpoint uses for the same data.
_REFERENCE_FIELDS = {
    "title": "title",
    "description": "description",
    "faviconUrl": "favicon",
    "imageUrl": "image",
    "date": "date",
}


def _text_content(content: Any, role: str) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str) for part in content
    ):
        return "\n".join(part["text"] for part in content)
    msg = "messages"
    raise UnsupportedParameterError(msg, PROVIDER_NAME, f"Liner accepts text content only, in a '{role}' message.")


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Reduce messages to the plain user and assistant text that Liner's agents accept."""
    converted: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role not in _ROLES:
            msg = "messages"
            raise UnsupportedParameterError(
                msg, PROVIDER_NAME, f"Liner accepts only user and assistant messages, not '{role}'."
            )
        converted.append({"role": role, "content": _text_content(message.get("content"), role)})
    if converted and converted[-1]["role"] != "user":
        msg = "messages"
        raise UnsupportedParameterError(msg, PROVIDER_NAME, "Liner needs the conversation to end with a user message.")
    return converted


def _convert_references(references: Any) -> list[dict[str, str]]:
    """Map Liner's references to citations, dropping empty fields and items without a URL."""
    citations: list[dict[str, str]] = []
    if not isinstance(references, list):
        return citations
    for reference in references:
        if not isinstance(reference, dict):
            continue
        url = reference.get("url")
        if not isinstance(url, str) or not url:
            continue
        citation = {"url": url}
        for source, target in _REFERENCE_FIELDS.items():
            value = reference.get(source)
            if isinstance(value, str) and value:
                citation[target] = value
        citations.append(citation)
    return citations


def _convert_response(response: dict[str, Any], model: str, response_id: str) -> ChatCompletion:
    answer = response.get("answer")
    if not isinstance(answer, str):
        msg = "Liner's response has no 'answer' string."
        raise ProviderError(msg, provider_name=PROVIDER_NAME)
    message = ChatCompletionMessage(
        role="assistant",
        content=answer,
        citations=_convert_references(response.get("references")),  # type: ignore[call-arg]
    )
    return ChatCompletion(
        id=response_id,
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[Choice(index=0, finish_reason="stop", message=message)],
    )


def _parse_event(line: str) -> dict[str, Any] | None:
    """Read one line of Liner's server-sent event stream, or None when it carries no event."""
    if not line.startswith("data:"):
        return None
    data = line.removeprefix("data:").strip()
    if not data or data == "[DONE]":
        return None
    event = json.loads(data)
    return event if isinstance(event, dict) else None


# The events of Liner's stream that become chunks; the rest are progress and metadata.
FORWARDED_EVENTS = frozenset({"text-delta", "data-search-references", "finish"})


def _raise_if_error_event(event: dict[str, Any]) -> None:
    """Raise the failure Liner reports inside a stream that already answered 200."""
    if event.get("type") != "data-error":
        return
    data = event.get("data")
    error = data if isinstance(data, dict) else {}
    code = error.get("code") if isinstance(error.get("code"), str) else None
    msg = f"Liner stream error: {error.get('message') or 'unknown error'}"
    raise ProviderError(msg, provider_name=PROVIDER_NAME, code=code)


def _convert_event(event: dict[str, Any], model: str, chunk_id: str, *, first: bool) -> ChatCompletionChunk:
    """Turn one of the FORWARDED_EVENTS into a chunk."""
    event_type = event.get("type")
    finish_reason: Literal["stop"] | None = None
    if event_type == "text-delta":
        delta = ChoiceDelta(content=str(event.get("delta", "")))
    elif event_type == "data-search-references":
        data = event.get("data")
        references = data.get("references") if isinstance(data, dict) else None
        delta = ChoiceDelta(citations=_convert_references(references))  # type: ignore[call-arg]
    else:
        delta = ChoiceDelta()
        finish_reason = "stop"
    if first:
        delta.role = "assistant"
    return ChatCompletionChunk(
        id=chunk_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
    )
