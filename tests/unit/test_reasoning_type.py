"""Tests for the Reasoning type's wire format and validation.

The Reasoning model wraps a string ``content`` field but serializes as a plain
string so that responses are wire-compatible with OpenAI-style clients that
expect ``delta.reasoning`` / ``message.reasoning`` to be a string.
"""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChoiceDelta,
    Reasoning,
)


def test_reasoning_serializes_as_plain_string() -> None:
    """A Reasoning model dumps as a string, not an object."""
    reasoning = Reasoning(content="thinking through the problem")
    # The serializer returns a string; mypy infers dict[str, Any] from BaseModel.model_dump,
    # so cast through Any to verify the runtime value.
    dumped: Any = reasoning.model_dump()
    assert dumped == "thinking through the problem"


def test_reasoning_json_dump_is_quoted_string() -> None:
    """model_dump_json produces a JSON string literal, not an object."""
    reasoning = Reasoning(content="hello")
    assert reasoning.model_dump_json() == '"hello"'


def test_reasoning_validates_from_plain_string() -> None:
    """Validation accepts a bare string (the wire format)."""
    reasoning = Reasoning.model_validate("a bare string")
    assert reasoning.content == "a bare string"


def test_reasoning_validates_from_object_form() -> None:
    """Validation still accepts the legacy ``{"content": str}`` object form."""
    reasoning = Reasoning.model_validate({"content": "object form"})
    assert reasoning.content == "object form"


def test_reasoning_attribute_access_preserved() -> None:
    """Typed attribute access remains the public Python API."""
    reasoning = Reasoning(content="abc")
    assert reasoning.content == "abc"


def test_choice_delta_reasoning_dumps_string() -> None:
    """The streaming delta serializes reasoning as a plain string."""
    delta = ChoiceDelta(reasoning=Reasoning(content="step one"))
    dumped = delta.model_dump()
    assert dumped["reasoning"] == "step one"


def test_chat_completion_chunk_reasoning_wire_format() -> None:
    """A full chunk dumps with delta.reasoning as a string, matching OpenAI."""
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "id",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning": {"content": "intermediate"}},
                    "finish_reason": None,
                }
            ],
        }
    )
    payload = json.loads(chunk.model_dump_json())
    assert payload["choices"][0]["delta"]["reasoning"] == "intermediate"


def test_chat_completion_chunk_accepts_string_reasoning_on_input() -> None:
    """Validation accepts ``delta.reasoning`` as a plain string (OpenAI wire form)."""
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "id",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning": "string form"},
                    "finish_reason": None,
                }
            ],
        }
    )
    assert chunk.choices[0].delta.reasoning is not None
    assert chunk.choices[0].delta.reasoning.content == "string form"


def test_chat_completion_message_reasoning_wire_format() -> None:
    """Non-streaming completion dumps message.reasoning as a string."""
    completion = ChatCompletion.model_validate(
        {
            "id": "id",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning": {"content": "because"},
                    },
                }
            ],
        }
    )
    payload = json.loads(completion.model_dump_json())
    assert payload["choices"][0]["message"]["reasoning"] == "because"


def test_chat_completion_message_accepts_string_reasoning_on_input() -> None:
    """Non-streaming validation accepts a bare string for message.reasoning."""
    completion = ChatCompletion.model_validate(
        {
            "id": "id",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning": "plain reasoning",
                    },
                }
            ],
        }
    )
    message = completion.choices[0].message
    assert message.reasoning is not None
    assert message.reasoning.content == "plain reasoning"


def test_reasoning_rejects_dict_without_content_key() -> None:
    """A dict that lacks the ``content`` key falls through and fails standard validation."""
    with pytest.raises(ValidationError):
        Reasoning.model_validate({"other_field": "value"})


def test_reasoning_rejects_dict_with_none_content() -> None:
    """A dict where ``content`` is None falls through and fails standard validation."""
    with pytest.raises(ValidationError):
        Reasoning.model_validate({"content": None})


@pytest.mark.parametrize("invalid_input", [123, None, [], 1.5, ("tuple",)])
def test_reasoning_rejects_non_string_non_dict_input(invalid_input: Any) -> None:
    """Non-string, non-dict inputs fall through and fail standard validation."""
    with pytest.raises(ValidationError):
        Reasoning.model_validate(invalid_input)


def test_reasoning_accepts_empty_string() -> None:
    """An empty string is valid reasoning content."""
    reasoning = Reasoning.model_validate("")
    assert reasoning.content == ""


def test_reasoning_coerces_non_string_content_in_dict() -> None:
    """A dict whose ``content`` value is not a string is coerced via ``str()``."""
    reasoning = Reasoning.model_validate({"content": 42})
    assert reasoning.content == "42"


def test_reasoning_roundtrip_through_json() -> None:
    """A chunk can be serialized to JSON and re-validated back to a Reasoning object."""
    original = ChatCompletionChunk.model_validate(
        {
            "id": "id",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning": {"content": "loop"}},
                    "finish_reason": None,
                }
            ],
        }
    )
    serialized = original.model_dump_json()
    revived = ChatCompletionChunk.model_validate_json(serialized)
    assert revived.choices[0].delta.reasoning is not None
    assert revived.choices[0].delta.reasoning.content == "loop"
