import json
from typing import Any

from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice


def _convert_instructor_response(instructor_response: Any, model: str, provider_name: str) -> ChatCompletion:
    """Convert instructor response to ChatCompletion format.

    Args:
        instructor_response: The response from instructor
        model: The model name used
        provider_name: The provider name (used in the response ID)

    Returns:
        ChatCompletion object with the structured response as JSON content

    """
    if hasattr(instructor_response, "model_dump"):
        content = json.dumps(instructor_response.model_dump())
    else:
        content = json.dumps(instructor_response)

    message = ChatCompletionMessage(
        role="assistant",
        content=content,
    )

    choice = Choice(
        finish_reason="stop",
        index=0,
        message=message,
    )

    return ChatCompletion(
        id=f"{provider_name}-instructor-response",
        choices=[choice],
        created=0,
        model=model,
        object="chat.completion",
    )
