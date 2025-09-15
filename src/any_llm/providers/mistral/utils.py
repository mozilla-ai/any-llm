import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from mistralai.models import AssistantMessageContent as MistralAssistantMessageContent
from mistralai.models import CompletionEvent
from mistralai.models import ModelList as MistralModelList
from mistralai.models import ReferenceChunk as MistralReferenceChunk
from mistralai.models import TextChunk as MistralTextChunk
from mistralai.models import ThinkChunk as MistralThinkChunk
from mistralai.models.chatcompletionresponse import ChatCompletionResponse as MistralChatCompletionResponse
from mistralai.models.toolcall import ToolCall as MistralToolCall
from mistralai.types.basemodel import Unset

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
    CreateEmbeddingResponse,
    Embedding,
    Function,
    Reasoning,
    Usage,
)
from any_llm.types.model import Model

if TYPE_CHECKING:
    from mistralai.models.embeddingresponse import EmbeddingResponse


def _convert_mistral_tool_calls_to_any_llm(
    tool_calls: list[MistralToolCall],
) -> list[ChatCompletionMessageToolCall] | None:
    """Convert Mistral tool calls to any-llm format."""
    if not tool_calls:
        return None

    any_llm_tool_calls: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] = []
    for tool_call in tool_calls:
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

        any_llm_tool_call = ChatCompletionMessageFunctionToolCall(
            id=tool_call.id,
            type="function",
            function=Function(
                name=tool_call.function.name,
                arguments=arguments,
            ),
        )
        any_llm_tool_calls.append(any_llm_tool_call)

    return any_llm_tool_calls


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
                            msg = f"Unsupported item type: {type(thinking_item)}"
                            raise ValueError(msg)
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
    """Create a ChatCompletion from Mistral response directly."""
    choices_out: list[Choice] = []

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

        tool_calls_final = (
            _convert_mistral_tool_calls_to_any_llm(message_data.tool_calls) if message_data.tool_calls else None
        )

        # if the content is none, see if it accidentally ended up in the reasoning content (aka <response>).
        # This is a bug in the mistral provider/model return
        if (
            content is None
            and reasoning_content
            and "<response>" in reasoning_content
            and "</response>" in reasoning_content
        ):
            content = reasoning_content.split("<response>")[1].split("</response>")[0]
            reasoning_content = reasoning_content.split("</response>")[0]

        message = ChatCompletionMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls_final,
            reasoning=Reasoning(content=reasoning_content) if reasoning_content else None,
        )

        choice = Choice(
            index=i,
            finish_reason=cast(
                "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']",
                choice_data.finish_reason,
            ),
            message=message,
        )
        choices_out.append(choice)

    usage = None
    if response_data.usage:
        usage = CompletionUsage(
            completion_tokens=response_data.usage.completion_tokens or 0,
            prompt_tokens=response_data.usage.prompt_tokens or 0,
            total_tokens=response_data.usage.total_tokens or 0,
        )

    return ChatCompletion(
        id=response_data.id,
        model=model,
        created=response_data.created,
        object="chat.completion",
        choices=choices_out,
        usage=usage,
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
                    if isinstance(part, MistralThinkChunk):
                        thinking_data = part.thinking
                        thinking_texts = []
                        for thinking_item in thinking_data:
                            if isinstance(thinking_item, MistralTextChunk):
                                thinking_texts.append(thinking_item.text)
                            elif isinstance(thinking_item, MistralReferenceChunk):
                                pass
                            else:
                                msg = f"Unsupported thinking item type: {type(thinking_item)}"
                                raise ValueError(msg)
                        if thinking_texts:
                            reasoning_content = "\n".join(thinking_texts)
                    elif isinstance(part, MistralTextChunk):
                        text_parts.append(part.text)
                content = "".join(text_parts) if text_parts else None
            else:
                content = str(choice.delta.content)

        role = None
        if choice.delta.role:
            role = cast("Literal['developer', 'system', 'user', 'assistant', 'tool']", choice.delta.role)

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


def _patch_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Patches messages for Mistral API compatibility.

    - Inserts an assistant message with "OK" content between a tool message and a user message.
    - Validates the message sequence to ensure correctness.
    """
    processed_msg: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        processed_msg.append(msg)
        if msg.get("role") == "tool":
            if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                # Mistral expects an assistant message after a tool message
                processed_msg.append({"role": "assistant", "content": "OK"})

    return processed_msg


def _convert_models_list(response: MistralModelList) -> Sequence[Model]:
    """Converts a Mistral ModelList to a list of Model objects."""
    models = []
    if response.data:
        for model_data in response.data:
            models.append(
                Model(
                    id=model_data.id,
                    created=model_data.created or 0,
                    object="model",
                    owned_by=model_data.owned_by or "mistral",
                )
            )
    return models
