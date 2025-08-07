import json
from typing import Any

from any_llm.types.completion import (
    CreateEmbeddingResponse,
    Embedding,
    Usage,
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    CompletionUsage,
    ChunkChoice,
    Reasoning,
)
from ollama import ChatResponse as OllamaChatResponse
from ollama import Message as OllamaMessage
from typing import Literal, cast
from datetime import datetime
from ollama import EmbedResponse
import uuid


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
        created = int(datetime.strptime(created_str, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())

    content = message.content

    role = None
    message_role = message.role
    if message_role:
        role = cast(Literal["developer", "system", "user", "assistant", "tool"], message_role)

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
            Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None, ollama_chunk.done_reason
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


def _create_response_dict_from_ollama_response(
    response: OllamaChatResponse,
) -> dict[str, Any]:
    """Convert an Ollama completion response to OpenAI format."""

    created_str = response.created_at
    if created_str is None:
        raise ValueError("Expected Ollama to provide a created_at timestamp")

    if "." in created_str and len(created_str.split(".")[1].rstrip("Z")) > 6:
        parts = created_str.split(".")
        microseconds = parts[1][:6]
        created_str = f"{parts[0]}.{microseconds}Z"
    created = int(datetime.strptime(created_str, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())

    prompt_tokens = response.prompt_eval_count or 0
    completion_tokens = response.eval_count or 0
    response_dict: dict[str, Any] = {
        "id": "chatcmpl-" + str(uuid.uuid4()),
        "model": response.model or "unknown",
        "created": created,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    response_message: OllamaMessage = response.message
    if not response_message or not isinstance(response_message, OllamaMessage):
        raise ValueError("Unexpected output from ollama")

    if response_message.tool_calls:
        tool_calls = []
        for tool_call in response_message.tool_calls:
            tool_calls.append(
                {
                    "id": str(uuid.uuid4()),
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": json.dumps(tool_call.function.arguments),
                    },
                    "type": "function",
                }
            )

        response_dict["choices"] = [
            {
                "message": {
                    "role": response_message.role,
                    "content": response_message.content,
                    "reasoning_content": response_message.thinking,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
                "index": 0,
            }
        ]
    else:
        response_dict["choices"] = [
            {
                "message": {
                    "role": response_message.role,
                    "content": response_message.content,
                    "reasoning_content": response_message.thinking,
                    "tool_calls": None,
                },
                "finish_reason": response.done_reason,
                "index": 0,
            }
        ]

    return response_dict
