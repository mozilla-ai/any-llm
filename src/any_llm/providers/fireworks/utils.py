# mypy: disable-error-code="union-attr"
from any_llm.types.completion import Reasoning
from any_llm.types.responses import Response


def extract_reasoning_from_response(response: Response) -> Response:
    """Extract <think> content from Fireworks response and set reasoning field.

    Fireworks Responses API may include reasoning content within <think></think> tags.
    This function extracts that content and moves it to the reasoning field.
    Args:
        response: The Response object to process

    Returns:
        The modified Response object with reasoning extracted
    """
    if response.reasoning:
        return response

    if not response.output or not response.output[-1].content:
        return response

    content_text = response.output[-1].content[0].text
    if "<think>" in content_text and "</think>" in content_text:
        reasoning = content_text.split("<think>")[1].split("</think>")[0].strip()
        if reasoning:
            response.reasoning = Reasoning(content=reasoning)  # type: ignore[assignment]
        response.output[-1].content[0].text = content_text.split("</think>")[1].strip()

    return response
