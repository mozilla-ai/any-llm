# mypy: disable-error-code="union-attr"
from typing import Any

from any_llm.types.completion import Reasoning
from any_llm.types.responses import Response
from any_llm.utils.reasoning import normalize_reasoning_from_provider_fields_and_xml_tags


def extract_reasoning_from_response(response: Response) -> Response:
    """Extract XML-tagged reasoning from a Fireworks response and set the reasoning field.

    Fireworks Responses API may include reasoning content within tags such as
    <think></think>. Parsing is delegated to
    normalize_reasoning_from_provider_fields_and_xml_tags so this provider keeps the same
    semantics as every other one: every tagged block is collected, and the surrounding
    content is preserved rather than truncated to whatever follows the last closing tag.

    Args:
        response: The Response object to process

    Returns:
        The modified Response object with reasoning extracted
    """
    if response.reasoning:
        return response

    if not response.output or not response.output[-1].content:
        return response

    message: dict[str, Any] = {"content": response.output[-1].content[0].text}
    normalize_reasoning_from_provider_fields_and_xml_tags(message)

    reasoning = message.get("reasoning")
    reasoning_text = reasoning.get("content") if isinstance(reasoning, dict) else None
    if reasoning_text:
        response.reasoning = Reasoning(content=reasoning_text)  # type: ignore[assignment]

    response.output[-1].content[0].text = message["content"]

    return response
