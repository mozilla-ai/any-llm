from typing import Any

from any_llm.types import ChatCompletion, Choice, CompletionUsage, ChatCompletionMessage


def _convert_response(response: dict[str, Any]) -> ChatCompletion:
    """Convert Watsonx response to OpenAI ChatCompletion format."""
    choice_data = response["choices"][0]
    message_data = choice_data["message"]

    # Create the message
    message = ChatCompletionMessage(
        content=message_data.get("content"),
        role=message_data.get("role", "assistant"),
        tool_calls=None,  # Watsonx doesn't seem to support tool calls in the aisuite implementation
    )

    # Create the choice
    choice = Choice(
        finish_reason=choice_data.get("finish_reason", "stop"),
        index=choice_data.get("index", 0),
        message=message,
    )

    # Create usage information (if available)
    usage = None
    if "usage" in response:
        usage_data = response["usage"]
        usage = CompletionUsage(
            completion_tokens=usage_data.get("completion_tokens", 0),
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

    # Build the final ChatCompletion object
    return ChatCompletion(
        id=response.get("id", ""),
        model=response.get("model", ""),
        object="chat.completion",
        created=response.get("created", 0),
        choices=[choice],
        usage=usage,
    )
