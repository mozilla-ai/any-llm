from typing import Any

import pytest

from any_llm.constants import REASONING_FIELD_NAMES
from any_llm.utils.reasoning import normalize_reasoning_from_provider_fields_and_xml_tags


@pytest.mark.parametrize(
    "tags",
    [
        ("think", "thinking"),
        ("thinking", "think"),
        ("think", "thinking", "think"),
        tuple(reversed(REASONING_FIELD_NAMES)),
    ],
)
def test_normalize_mixed_reasoning_tags_keeps_textual_order(tags: tuple[str, ...]) -> None:
    blocks = [f"<{tag}>step {index}</{tag}>" for index, tag in enumerate(tags)]
    message: dict[str, Any] = {"content": "  prefix " + " between ".join(blocks) + " suffix  "}

    normalize_reasoning_from_provider_fields_and_xml_tags(message)

    assert message["reasoning"] == {"content": "\n".join(f"step {index}" for index in range(len(tags)))}
    assert message["content"] == "prefix " + " between " * (len(tags) - 1) + " suffix"


@pytest.mark.parametrize(
    ("content", "expected_content", "expected_reasoning"),
    [
        ("  no tags  ", "  no tags  ", None),
        ("  <unknown>keep</unknown>  ", "  <unknown>keep</unknown>  ", None),
        ("  <think>keep</thinking>  ", "  <think>keep</thinking>  ", None),
        ("  <think></think>answer  ", "answer", ""),
        ("  <think></think><thinking></thinking>answer  ", "answer", ""),
        ("  <think>  first\n</think> between <thinking> second </thinking>  ", "between", "  first\n\n second "),
    ],
)
def test_normalize_reasoning_preserves_empty_and_whitespace_semantics(
    content: str, expected_content: str, expected_reasoning: str | None
) -> None:
    message: dict[str, Any] = {"content": content}

    normalize_reasoning_from_provider_fields_and_xml_tags(message)

    assert message["content"] == expected_content
    if expected_reasoning is None:
        assert "reasoning" not in message
    else:
        assert message["reasoning"] == {"content": expected_reasoning}


def test_normalize_reasoning_preserves_provider_field_precedence() -> None:
    message: dict[str, Any] = {
        "content": "<think>first</think><thinking>second</thinking>answer",
        "reasoning_content": "preferred field",
        "thinking": "other field",
        "reasoning": "fallback field",
    }

    normalize_reasoning_from_provider_fields_and_xml_tags(message)

    assert message["reasoning"] == {"content": "preferred field\nfirst\nsecond"}
    assert message["content"] == "answer"


def test_normalize_reasoning_preserves_already_normalized_reasoning() -> None:
    message: dict[str, Any] = {
        "content": "<think>first</think><thinking>second</thinking>answer",
        "reasoning": {"content": "already normalized"},
    }
    original = message.copy()

    normalize_reasoning_from_provider_fields_and_xml_tags(message)

    assert message == original
