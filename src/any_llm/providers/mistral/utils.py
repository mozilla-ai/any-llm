import json

try:
    from mistralai.models import CompletionEvent
    from mistralai.models.embeddingresponse import EmbeddingResponse
except ImportError:
    msg = "mistralai is not installed. Please install it with `pip install any-llm-sdk[mistral]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types import CreateEmbeddingResponse


def _create_openai_chunk_from_mistral_chunk(event: CompletionEvent) -> ChatCompletionChunk:
    """Convert a Mistral CompletionEvent to OpenAI ChatCompletionChunk format."""
    from typing import Literal, cast

    from openai.types.chat.chat_completion_chunk import (
        Choice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from openai.types.completion_usage import CompletionUsage

    chunk = event.data

    # Convert choices
    openai_choices = []
    for choice in chunk.choices:
        # Convert delta
        content = None
        if choice.delta.content:
            # Handle complex content types by converting to string if needed
            if isinstance(choice.delta.content, str):
                content = choice.delta.content
            elif isinstance(choice.delta.content, list):
                # Extract text content from complex content types
                text_parts = []
                for part in choice.delta.content:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(str(part.text))
                content = "".join(text_parts) if text_parts else None
            else:
                content = str(choice.delta.content)

        # Handle role with proper type casting
        role = None
        if choice.delta.role:
            role = cast(Literal["developer", "system", "user", "assistant", "tool"], choice.delta.role)

        delta = ChoiceDelta(content=content, role=role)

        # Convert tool calls if present
        if choice.delta.tool_calls:
            tool_calls = []
            for tool_call in choice.delta.tool_calls:
                # Handle index with proper default
                index = tool_call.index if tool_call.index is not None else 0

                # Handle function arguments conversion
                arguments = None
                if tool_call.function:
                    func_args = tool_call.function.arguments
                    if isinstance(func_args, dict):
                        arguments = json.dumps(func_args)
                    elif isinstance(func_args, str):
                        arguments = func_args
                    else:
                        arguments = str(func_args) if func_args is not None else None

                openai_tool_call = ChoiceDeltaToolCall(
                    index=index,
                    id=tool_call.id,
                    type="function",
                    function=ChoiceDeltaToolCallFunction(
                        name=tool_call.function.name if tool_call.function else None,
                        arguments=arguments,
                    )
                    if tool_call.function
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
    from openai.types.embedding import Embedding
    from openai.types.create_embedding_response import Usage

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
