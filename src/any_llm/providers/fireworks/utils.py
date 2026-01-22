# mypy: disable-error-code="union-attr"
from openai.types.responses import Response


def extract_reasoning_from_response(response: Response) -> Response:
    """Extract <think> content from Fireworks response.

    Fireworks Responses API may include reasoning content within <think></think> tags.
    This function extracts that content and cleans the output text.

    Note: In OpenResponses, reasoning content is part of output items, not a separate
    top-level field. This function primarily cleans the text output.

    Args:
        response: The Response object to process

    Returns:
        The modified Response object with reasoning extracted from text
    """
    if not response.output:
        return response

    last_output = response.output[-1]
    content_list = getattr(last_output, "content", None)
    if not content_list:
        return response

    first_content = content_list[0]
    content_text = getattr(first_content, "text", "")
    if not content_text:
        return response

    if "<think>" in content_text and "</think>" in content_text:
        # Extract reasoning content (for logging/debugging purposes)
        reasoning_text = content_text.split("<think>")[1].split("</think>")[0].strip()  # noqa: F841
        # Clean the output text by removing the think tags
        cleaned_text = content_text.split("</think>")[1].strip()
        # Update the content text
        if hasattr(first_content, "text"):
            object.__setattr__(first_content, "text", cleaned_text)

    return response
