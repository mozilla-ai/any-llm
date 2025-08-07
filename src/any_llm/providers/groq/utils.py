from typing import cast, Literal, Any

from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    CompletionUsage,
    ChunkChoice,
    Reasoning,
)

from groq.types.chat import ChatCompletionChunk as GroqChatCompletionChunk
from groq.types.chat import ChatCompletion as GroqChatCompletion


def _create_response_dict_from_groq_response(
    response: GroqChatCompletion,
) -> dict[str, Any]:
    """Convert a Groq completion response to OpenAI format."""

    response_dict: dict[str, Any] = {
        "id": response.id,
        "model": response.model,
        "created": response.created,
        "object": "chat.completion",
    }

    # Handle usage
    if response.usage:
        response_dict["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    # Handle choices
    choices = []
    for choice in response.choices:
        choice_dict: dict[str, Any] = {
            "index": choice.index,
            "finish_reason": choice.finish_reason,
        }

        # Handle message
        message = choice.message
        message_dict: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }

        # Handle tool calls if present
        if message.tool_calls:
            tool_calls = []
            for tool_call in message.tool_calls:
                tool_call_dict = {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                tool_calls.append(tool_call_dict)
            message_dict["tool_calls"] = tool_calls
        else:
            message_dict["tool_calls"] = None

        # Handle reasoning if present (Groq specific)
        if message.reasoning:
            message_dict["reasoning_content"] = message.reasoning

        choice_dict["message"] = message_dict
        choices.append(choice_dict)

    response_dict["choices"] = choices

    return response_dict


def _create_openai_chunk_from_groq_chunk(groq_chunk: GroqChatCompletionChunk) -> ChatCompletionChunk:
    """Convert a Groq streaming chunk to OpenAI ChatCompletionChunk format."""

    choice_data = groq_chunk.choices[0]
    delta_data = choice_data.delta

    delta = ChoiceDelta(
        content=delta_data.content,
        reasoning=Reasoning(content=delta_data.reasoning) if delta_data.reasoning else None,
        role=cast(Literal["developer", "system", "user", "assistant", "tool"] | None, delta_data.role),
    )

    if delta_data.tool_calls:
        openai_tool_calls = []
        for tool_call in delta_data.tool_calls:
            openai_tool_call = ChoiceDeltaToolCall(
                index=tool_call.index if tool_call.index is not None else 0,
                id=tool_call.id,
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=tool_call.function.name if tool_call.function else None,
                    arguments=tool_call.function.arguments if tool_call.function else None,
                )
                if tool_call.function
                else None,
            )
            openai_tool_calls.append(openai_tool_call)
        delta.tool_calls = openai_tool_calls
    else:
        delta.tool_calls = None

    choice = ChunkChoice(
        index=choice_data.index,
        delta=delta,
        finish_reason=choice_data.finish_reason,
    )

    usage = None
    usage_data = groq_chunk.usage
    if usage_data:
        usage = CompletionUsage(
            prompt_tokens=usage_data.prompt_tokens,
            completion_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens,
        )

    return ChatCompletionChunk(
        id=groq_chunk.id,
        choices=[choice],
        created=groq_chunk.created,
        model=groq_chunk.model,
        object="chat.completion.chunk",
        usage=usage,
    )
