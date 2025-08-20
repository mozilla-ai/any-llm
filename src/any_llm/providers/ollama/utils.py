import json
import uuid
from datetime import UTC, datetime
from typing import Any, Literal, cast

from ollama import ChatResponse as OllamaChatResponse
from ollama import EmbedResponse
from ollama import ListResponse as OllamaListResponse
from ollama import Message as OllamaMessage

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


def _create_openai_embedding_response_from_ollama(
    ollama_response: EmbedResponse,
) -> CreateEmbeddingResponse:
    """Convert an Ollama embedding response to OpenAI CreateEmbeddingResponse format."""

    openai_embeddings = []

    for index, embedding_vector in enumerate(ollama_response.embeddings):
        openai_embedding = Embedding(embedding=list(embedding_vector), index=index, object="embedding")
        openai_embeddings.append(openai_embedding)

    prompt_tokens = ollama_response.prompt_eval_count
    usage = Usage(
        prompt_tokens=prompt_tokens or 0,
        total_tokens=prompt_tokens or 0,
    )

    return CreateEmbeddingResponse(
        data=openai_embeddings,
        model=ollama_response.model or "unknown",
        object="list",
        usage=usage,
    )


def _create_openai_chunk_from_ollama_chunk(ollama_chunk: OllamaChatResponse) -> ChatCompletionChunk:
    """Convert an Ollama streaming chunk to OpenAI ChatCompletionChunk format."""

    message = ollama_chunk.message
    created_str = ollama_chunk.created_at
    created = 0
    if created_str:
        if "." in created_str and len(created_str.split(".")[1].rstrip("Z")) > 6:
            parts = created_str.split(".")
            microseconds = parts[1][:6]
            created_str = f"{parts[0]}.{microseconds}Z"
        created = int(datetime.strptime(created_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC).timestamp())

    content = message.content

    role = None
    message_role = message.role
    if message_role:
        role = cast("Literal['developer', 'system', 'user', 'assistant', 'tool']", message_role)

    delta = ChoiceDelta(
        content=content, role=role, reasoning=Reasoning(content=message.thinking) if message.thinking else None
    )

    tool_calls = message.tool_calls
    if tool_calls:
        openai_tool_calls = []
        for tool_call in tool_calls:
            arguments = None
            func = tool_call.function
            func_args = func.arguments

            if isinstance(func_args, dict):
                arguments = json.dumps(func_args)

            openai_tool_call = ChoiceDeltaToolCall(
                index=0,
                id=str(uuid.uuid4()),
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=func.name,
                    arguments=arguments,
                ),
            )
            openai_tool_calls.append(openai_tool_call)
        delta.tool_calls = openai_tool_calls

    choice = ChunkChoice(
        index=0,
        delta=delta,
        finish_reason=cast(
            "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'] | None",
            ollama_chunk.done_reason,
        ),
    )

    usage = None
    prompt_tokens = ollama_chunk.prompt_eval_count
    completion_tokens = ollama_chunk.eval_count
    if prompt_tokens or completion_tokens:
        usage = CompletionUsage(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
        )

    return ChatCompletionChunk(
        id=f"chatcmpl-{uuid.uuid4()}",
        choices=[choice],
        created=created,
        model=ollama_chunk.model or "unknown",
        object="chat.completion.chunk",
        usage=usage,
    )


def _create_chat_completion_from_ollama_response(response: OllamaChatResponse) -> ChatCompletion:
    """Convert an Ollama completion response directly to an OpenAI-compatible ChatCompletion."""

    created_str = response.created_at
    if created_str is None:
        msg = "Expected Ollama to provide a created_at timestamp"
        raise ValueError(msg)

    if "." in created_str and len(created_str.split(".")[1].rstrip("Z")) > 6:
        parts = created_str.split(".")
        microseconds = parts[1][:6]
        created_str = f"{parts[0]}.{microseconds}Z"
    created = int(datetime.strptime(created_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC).timestamp())

    prompt_tokens = response.prompt_eval_count or 0
    completion_tokens = response.eval_count or 0

    response_message: OllamaMessage = response.message
    if not response_message or not isinstance(response_message, OllamaMessage):
        msg = "Unexpected output from ollama"
        raise ValueError(msg)

    openai_tool_calls: list[ChatCompletionMessageToolCall] | None = None
    if response_message.tool_calls:
        openai_tool_calls = []
        for tool_call in response_message.tool_calls:
            raw_arguments = tool_call.function.arguments
            if isinstance(raw_arguments, dict):
                arguments_str = json.dumps(raw_arguments)
            elif isinstance(raw_arguments, str):
                arguments_str = raw_arguments
            else:
                arguments_str = json.dumps(raw_arguments) if raw_arguments is not None else "{}"
            openai_tool_calls.append(
                ChatCompletionMessageFunctionToolCall(
                    id=str(uuid.uuid4()),
                    type="function",
                    function=Function(
                        name=tool_call.function.name,
                        arguments=arguments_str,
                    ),
                )
            )
    if not response_message.thinking and response_message.content:
        # If it didn't come out right from ollama, also look for it in the content between <think> and </think>
        if "<think>" in response_message.content and "</think>" in response_message.content:
            response_message.thinking = response_message.content.split("<think>")[1].split("</think>")[0]
            # remove it from the content
            response_message.content = response_message.content.split("<think>")[0]

    message = ChatCompletionMessage(
        role="assistant",
        content=response_message.content,
        tool_calls=openai_tool_calls,
        reasoning=Reasoning(content=response_message.thinking) if response_message.thinking else None,
    )

    finish_reason: Any = "tool_calls" if openai_tool_calls else response.done_reason

    choice = Choice(index=0, finish_reason=finish_reason, message=message)

    usage = CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    return ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4()}",
        model=response.model or "unknown",
        created=created,
        object="chat.completion",
        choices=[choice],
        usage=usage,
    )


def _convert_models_list(models_list: OllamaListResponse) -> list[Model]:
    models = models_list.models
    return [
        Model(
            id=model.model or "Unknown",
            object="model",
            created=int(model.modified_at.timestamp()) if model.modified_at else 0,
            owned_by="ollama",
        )
        for model in models
    ]
