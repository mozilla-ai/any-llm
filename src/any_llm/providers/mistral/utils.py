try:
    from mistralai.models import CompletionEvent
except ImportError:
    msg = "mistralai is not installed. Please install it with `pip install any-llm-sdk[mistral]`"
    raise ImportError(msg)

from any_llm.types import ChatCompletionChunk


def _create_openai_chunk_from_mistral_chunk(event: CompletionEvent) -> ChatCompletionChunk:
    """Convert a Mistral CompletionEvent to OpenAI ChatCompletionChunk format."""
    from typing import Literal, cast
    import json

    from openai.types.chat.chat_completion_chunk import (
        Choice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from any_llm.types import CompletionUsage

    chunk = event.data

    # Convert choices
    openai_choices = []
    for choice in chunk.choices:
        # Convert delta
        content = None
        if hasattr(choice.delta, "content") and choice.delta.content:
            # Handle complex content types by converting to string if needed
            if isinstance(choice.delta.content, str):
                content = choice.delta.content
            else:
                content = str(choice.delta.content)

        # Handle role with proper type casting
        role = None
        if hasattr(choice.delta, "role") and choice.delta.role:
            # Cast to one of the expected literal types
            role_str = choice.delta.role
            if role_str in ["developer", "system", "user", "assistant", "tool"]:
                role = cast(Literal["developer", "system", "user", "assistant", "tool"], role_str)

        delta = ChoiceDelta(content=content, role=role)

        # Convert tool calls if present
        if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
            tool_calls = []
            for tool_call in choice.delta.tool_calls:
                # Handle index with proper default
                index = tool_call.index if hasattr(tool_call, "index") and tool_call.index is not None else 0

                # Handle function arguments conversion
                arguments = None
                if hasattr(tool_call, "function") and tool_call.function:
                    if hasattr(tool_call.function, "arguments") and tool_call.function.arguments is not None:
                        if isinstance(tool_call.function.arguments, dict):
                            arguments = json.dumps(tool_call.function.arguments)
                        else:
                            arguments = tool_call.function.arguments

                openai_tool_call = ChoiceDeltaToolCall(
                    index=index,
                    id=tool_call.id if hasattr(tool_call, "id") else None,
                    type="function",
                    function=ChoiceDeltaToolCallFunction(
                        name=tool_call.function.name if hasattr(tool_call, "function") and tool_call.function else None,
                        arguments=arguments,
                    )
                    if hasattr(tool_call, "function") and tool_call.function
                    else None,
                )
                tool_calls.append(openai_tool_call)
            delta.tool_calls = tool_calls

        openai_choice = Choice(
            index=choice.index,
            delta=delta,
            finish_reason=choice.finish_reason,  # type: ignore[arg-type]
        )
        openai_choices.append(openai_choice)

    # Convert usage if present
    usage = None
    if chunk.usage:
        usage = CompletionUsage(
            prompt_tokens=chunk.usage.prompt_tokens if hasattr(chunk.usage, "prompt_tokens") else 0,
            completion_tokens=chunk.usage.completion_tokens if hasattr(chunk.usage, "completion_tokens") else 0,
            total_tokens=chunk.usage.total_tokens if hasattr(chunk.usage, "total_tokens") else 0,
        )

    return ChatCompletionChunk(
        id=chunk.id,
        choices=openai_choices,
        created=chunk.created or 0,
        model=chunk.model,
        object="chat.completion.chunk",
        usage=usage,
    )
