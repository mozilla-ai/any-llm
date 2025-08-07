import json

try:
    from mistralai.models import CompletionEvent
    from mistralai.models.embeddingresponse import EmbeddingResponse
    from mistralai.models.chatcompletionresponse import ChatCompletionResponse as MistralChatCompletionResponse
    from mistralai.models import AssistantMessageContent as MistralAssistantMessageContent
    from mistralai.models import ThinkChunk as MistralThinkChunk
    from mistralai.models import TextChunk as MistralTextChunk
    from mistralai.models import ReferenceChunk as MistralReferenceChunk
    from mistralai.models.toolcall import ToolCall as MistralToolCall
    from mistralai.types.basemodel import Unset
except ImportError as exc:
    msg = "mistralai is not installed. Please install it with `pip install any-llm-sdk[mistral]`"
    raise ImportError(msg) from exc

from any_llm.types.completion import (
    ChatCompletionChunk,
    CreateEmbeddingResponse,
    ChatCompletion,
    Reasoning,
    CompletionUsage,
    Embedding,
    Usage,
    ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    Function,
)
from any_llm.providers.helpers import (
    create_completion_from_response,
)
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from typing import Literal, cast, Any


def _convert_mistral_tool_calls_to_any_llm(
    tool_calls: list[MistralToolCall],
) -> list[ChatCompletionMessageToolCall] | None:
    """Convert Mistral tool calls to any-llm format."""
    if not tool_calls:
        return None

    any_llm_tool_calls = []
    for tool_call in tool_calls:
        arguments = ""
        if tool_call.function and tool_call.function.arguments:
            if isinstance(tool_call.function.arguments, dict):
                arguments = json.dumps(tool_call.function.arguments)
            elif isinstance(tool_call.function.arguments, str):
                arguments = tool_call.function.arguments
            else:
                arguments = str(tool_call.function.arguments)

        # Skip tool calls without required fields
        if not tool_call.id or not tool_call.function or not tool_call.function.name:
            continue

        any_llm_tool_call = ChatCompletionMessageFunctionToolCall(
            id=tool_call.id,
            type="function",
            function=Function(
                name=tool_call.function.name,
                arguments=arguments,
            ),
        )
        any_llm_tool_calls.append(any_llm_tool_call)

    return any_llm_tool_calls  # type: ignore[return-value]


def _convert_mistral_streaming_tool_calls_to_any_llm(
    tool_calls: list[MistralToolCall],
) -> list[ChoiceDeltaToolCall] | None:
    """Convert Mistral streaming tool calls to any-llm format."""
    if not tool_calls:
        return None

    any_llm_tool_calls = []
    for tool_call in tool_calls:
        index = tool_call.index if tool_call.index is not None else 0

        arguments = ""
        if tool_call.function and tool_call.function.arguments:
            if isinstance(tool_call.function.arguments, dict):
                arguments = json.dumps(tool_call.function.arguments)
            elif isinstance(tool_call.function.arguments, str):
                arguments = tool_call.function.arguments
            else:
                arguments = str(tool_call.function.arguments)

        if not tool_call.id or not tool_call.function or not tool_call.function.name:
            continue

        openai_tool_call = ChoiceDeltaToolCall(
            index=index,
            id=tool_call.id,
            type="function",
            function=ChoiceDeltaToolCallFunction(
                name=tool_call.function.name,
                arguments=arguments,
            ),
        )
        any_llm_tool_calls.append(openai_tool_call)

    return any_llm_tool_calls


def _extract_mistral_content_and_reasoning(
    content_data: MistralAssistantMessageContent,
) -> tuple[str | None, str | None]:
    """
    Extract text content and reasoning from Mistral's content structure.

    Mistral returns content as an array of objects, where reasoning is in a 'thinking' object.
    """

    text_parts = []
    reasoning_content = None

    for item in content_data:
        if isinstance(item, str):
            text_parts.append(item)
        else:
            if isinstance(item, MistralThinkChunk):
                thinking_data = item.thinking
                if isinstance(thinking_data, list):
                    thinking_texts = []
                    for thinking_item in thinking_data:
                        if isinstance(thinking_item, MistralTextChunk):
                            thinking_texts.append(thinking_item.text)
                        elif isinstance(thinking_item, MistralReferenceChunk):
                            pass
                        else:
                            raise ValueError(f"Unsupported item type: {type(thinking_item)}")
                    if thinking_texts:
                        reasoning_content = "\n".join(thinking_texts)
                elif isinstance(thinking_data, str):
                    reasoning_content = thinking_data
            elif isinstance(item, MistralTextChunk):
                text_parts.append(item.text)

    content = "".join(text_parts) if text_parts else None
    return content, reasoning_content


def _create_mistral_completion_from_response(
    response_data: MistralChatCompletionResponse, model: str
) -> ChatCompletion:
    """Create a ChatCompletion from Mistral response via normalization helper."""
    choices_norm: list[dict[str, Any]] = []

    for i, choice_data in enumerate(response_data.choices):
        message_data = choice_data.message

        if message_data.content:
            content, reasoning_content = _extract_mistral_content_and_reasoning(message_data.content)
        else:
            content = None
            reasoning_content = None

        tool_calls_list: list[dict[str, Any]] | None = None
        if message_data.tool_calls is not None and not isinstance(message_data.tool_calls, Unset):
            tool_calls_list = []
            for tc in message_data.tool_calls:
                args: Any = tc.function.arguments if tc.function else None
                if isinstance(args, dict):
                    args = json.dumps(args)
                elif args is not None and not isinstance(args, str):
                    args = str(args)
                tool_calls_list.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name if tc.function else None,
                            "arguments": args,
                        },
                    }
                )

        choices_norm.append(
            {
                "message": {
                    "role": message_data.role,
                    "content": content,
                    "tool_calls": tool_calls_list,
                    "reasoning_content": reasoning_content,
                },
                "finish_reason": choice_data.finish_reason,
                "index": i,
            }
        )

    usage_norm: dict[str, Any] | None = None
    if response_data.usage:
        usage_norm = {
            "completion_tokens": response_data.usage.completion_tokens or 0,
            "prompt_tokens": response_data.usage.prompt_tokens or 0,
            "total_tokens": response_data.usage.total_tokens or 0,
        }

    normalized: dict[str, Any] = {
        "id": response_data.id,
        "model": model,
        "created": response_data.created,
        "choices": choices_norm,
        "usage": usage_norm,
    }

    return create_completion_from_response(
        response_data=normalized,
        model=model,
        provider_name="mistral",
    )


def _create_openai_chunk_from_mistral_chunk(event: CompletionEvent) -> ChatCompletionChunk:
    """Convert a Mistral CompletionEvent to OpenAI ChatCompletionChunk format."""
    chunk = event.data

    openai_choices = []
    for choice in chunk.choices:
        content = None
        reasoning_content = None

        if choice.delta.content:
            if isinstance(choice.delta.content, str):
                content = choice.delta.content
            elif isinstance(choice.delta.content, list):
                text_parts = []
                for part in choice.delta.content:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(str(part.text))
                    elif isinstance(part, dict):
                        # Handle reasoning content in streaming
                        if part.type == "thinking":
                            thinking_data = part.thinking
                            if isinstance(thinking_data, list):
                                thinking_texts = []
                                for thinking_item in thinking_data:
                                    if isinstance(thinking_item, dict) and thinking_item.type == "text":
                                        thinking_texts.append(thinking_item.text)
                                if thinking_texts:
                                    reasoning_content = "\n".join(thinking_texts)
                            elif isinstance(thinking_data, str):
                                reasoning_content = thinking_data
                        elif part.type == "text":
                            text_parts.append(part.text)
                content = "".join(text_parts) if text_parts else None
            else:
                content = str(choice.delta.content)

        role = None
        if choice.delta.role:
            role = cast(Literal["developer", "system", "user", "assistant", "tool"], choice.delta.role)

        reasoning = None
        if reasoning_content:
            reasoning = Reasoning(content=reasoning_content)

        delta = ChoiceDelta(content=content, role=role, reasoning=reasoning)

        delta.tool_calls = None
        if choice.delta.tool_calls is not None and not isinstance(choice.delta.tool_calls, Unset):
            delta.tool_calls = _convert_mistral_streaming_tool_calls_to_any_llm(choice.delta.tool_calls)

        openai_choice = ChunkChoice(
            index=choice.index,
            delta=delta,
            finish_reason=choice.finish_reason,  # type: ignore[arg-type]
        )
        openai_choices.append(openai_choice)

    usage = None
    if chunk.usage:
        usage = CompletionUsage(
            prompt_tokens=chunk.usage.prompt_tokens or 0,
            completion_tokens=chunk.usage.completion_tokens or 0,
            total_tokens=chunk.usage.total_tokens or 0,
        )

    return ChatCompletionChunk(
        id=chunk.id,
        choices=openai_choices,
        created=chunk.created or 0,
        model=chunk.model,
        object="chat.completion.chunk",
        usage=usage,
    )


def _create_openai_embedding_response_from_mistral(
    mistral_response: "EmbeddingResponse",
) -> "CreateEmbeddingResponse":
    """Convert a Mistral EmbeddingResponse to OpenAI CreateEmbeddingResponse format."""

    openai_embeddings = []
    for embedding_data in mistral_response.data:
        embedding_vector = embedding_data.embedding or []

        openai_embedding = Embedding(embedding=embedding_vector, index=embedding_data.index or 0, object="embedding")
        openai_embeddings.append(openai_embedding)

    usage = Usage(
        prompt_tokens=mistral_response.usage.prompt_tokens or 0,
        total_tokens=mistral_response.usage.total_tokens or 0,
    )

    return CreateEmbeddingResponse(
        data=openai_embeddings,
        model=mistral_response.model,
        object="list",
        usage=usage,
    )
