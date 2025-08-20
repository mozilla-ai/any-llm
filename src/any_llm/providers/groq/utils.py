from typing import Literal, cast

from groq.types import ModelListResponse as GroqModelListResponse
from groq.types.chat import ChatCompletion as GroqChatCompletion
from groq.types.chat import ChatCompletionChunk as GroqChatCompletionChunk

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    CompletionUsage,
    Function,
    Reasoning,
)
from any_llm.types.model import Model


def to_chat_completion(response: GroqChatCompletion) -> ChatCompletion:
    """Convert Groq ChatCompletion into our ChatCompletion type directly."""

    usage = None
    if response.usage:
        usage = CompletionUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    choices: list[Choice] = []
    for choice in response.choices:
        message = choice.message

        tool_calls: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] | None = None
        if message.tool_calls:
            tool_calls_list: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] = []
            for tool_call in message.tool_calls:
                arguments = tool_call.function.arguments
                if not isinstance(arguments, str):
                    # Ensure arguments is a string
                    arguments = str(arguments)
                tool_calls_list.append(
                    ChatCompletionMessageFunctionToolCall(
                        id=tool_call.id,
                        type="function",
                        function=Function(
                            name=tool_call.function.name,
                            arguments=arguments,
                        ),
                    )
                )
            tool_calls = tool_calls_list

        msg = ChatCompletionMessage(
            role=cast("Literal['assistant']", "assistant"),
            content=message.content,
            tool_calls=tool_calls,
            reasoning=Reasoning(content=cast("str", message.reasoning))
            if getattr(message, "reasoning", None)
            else None,
        )

        choices.append(
            Choice(
                index=choice.index,
                finish_reason=cast(
                    "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']",
                    choice.finish_reason or "stop",
                ),
                message=msg,
            )
        )

    return ChatCompletion(
        id=response.id,
        model=response.model,
        created=response.created,
        object="chat.completion",
        choices=choices,
        usage=usage,
    )


def _create_openai_chunk_from_groq_chunk(groq_chunk: GroqChatCompletionChunk) -> ChatCompletionChunk:
    """Convert a Groq streaming chunk to OpenAI ChatCompletionChunk format."""

    choice_data = groq_chunk.choices[0]
    delta_data = choice_data.delta

    delta = ChoiceDelta(
        content=delta_data.content,
        reasoning=Reasoning(content=delta_data.reasoning) if delta_data.reasoning else None,
        role=cast("Literal['developer', 'system', 'user', 'assistant', 'tool'] | None", delta_data.role),
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


def _convert_models_list(models_list: GroqModelListResponse) -> list[Model]:
    return [Model(id=model.id, object="model", created=model.created, owned_by="groq") for model in models_list.data]
