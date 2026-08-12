from collections.abc import Sequence
from typing import Any

from cohere import V2ChatResponse
from cohere.types import ListModelsResponse as CohereListModelsResponse

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    Function,
    Reasoning,
    Usage,
)
from any_llm.types.model import Model
from any_llm.types.rerank import RerankMeta, RerankResponse, RerankResult, RerankUsage


def _patch_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Patches messages for Cohere API compatibility.

    - Removes the 'name' field from tool messages.
    - Converts 'content' to 'tool_plan' in assistant messages with tool_calls.
    - Validates the message sequence.
    """
    patched_messages = []
    for i, message in enumerate(messages):
        patched_message = message.copy()
        if patched_message.get("role") == "tool":
            # Walk backwards past sibling tool messages to find the assistant message
            j = i - 1
            while j >= 0 and messages[j].get("role") == "tool":
                j -= 1
            if j < 0 or messages[j].get("role") != "assistant":
                msg = "A tool message must be preceded by an assistant message with tool_calls."
                raise ValueError(msg)
            patched_message.pop("name", None)
        if patched_message.get("role") == "assistant" and patched_message.get("tool_calls"):
            patched_message["tool_plan"] = patched_message.pop("content")
        patched_messages.append(patched_message)
    return patched_messages


def _create_openai_chunk_from_cohere_chunk(chunk: Any) -> ChatCompletionChunk:
    """Convert Cohere streaming chunk to OpenAI ChatCompletionChunk format."""
    chunk_dict: dict[str, Any] = {
        "id": f"chatcmpl-{hash(str(chunk))}",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "cohere-model",
        "choices": [],
        "usage": None,
    }

    delta: dict[str, Any] = {}
    finish_reason = None

    chunk_type = getattr(chunk, "type", None)

    if chunk_type == "content-delta":
        if (
            hasattr(chunk, "delta")
            and chunk.delta
            and hasattr(chunk.delta, "message")
            and chunk.delta.message
            and hasattr(chunk.delta.message, "content")
            and chunk.delta.message.content
        ):
            content_obj = chunk.delta.message.content
            if hasattr(content_obj, "text") and content_obj.text:
                delta["content"] = content_obj.text
            elif hasattr(content_obj, "thinking") and content_obj.thinking:
                delta["reasoning"] = {"content": content_obj.thinking}

    elif chunk_type == "tool-call-start":
        if (
            hasattr(chunk, "delta")
            and chunk.delta
            and hasattr(chunk.delta, "message")
            and chunk.delta.message
            and hasattr(chunk.delta.message, "tool_calls")
            and chunk.delta.message.tool_calls
        ):
            tool_call = chunk.delta.message.tool_calls
            delta["tool_calls"] = [
                {
                    "index": chunk.index or 0,
                    "id": getattr(tool_call, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(tool_call.function, "name", "")
                        if hasattr(tool_call, "function") and tool_call.function
                        else "",
                        "arguments": "",
                    },
                }
            ]

    elif chunk_type == "tool-call-delta":
        if (
            hasattr(chunk, "delta")
            and chunk.delta
            and hasattr(chunk.delta, "message")
            and chunk.delta.message
            and hasattr(chunk.delta.message, "tool_calls")
            and chunk.delta.message.tool_calls
            and hasattr(chunk.delta.message.tool_calls, "function")
            and chunk.delta.message.tool_calls.function
        ):
            delta["tool_calls"] = [
                {
                    "index": chunk.index or 0,
                    "function": {
                        "arguments": getattr(chunk.delta.message.tool_calls.function, "arguments", ""),
                    },
                }
            ]

    elif chunk_type == "tool-call-end":
        finish_reason = "tool_calls"

    elif chunk_type == "message-end":
        finish_reason = "stop"

        if (
            hasattr(chunk, "delta")
            and chunk.delta
            and hasattr(chunk.delta, "usage")
            and chunk.delta.usage
            and hasattr(chunk.delta.usage, "tokens")
            and chunk.delta.usage.tokens
        ):
            chunk_dict["usage"] = {
                "prompt_tokens": int(getattr(chunk.delta.usage.tokens, "input_tokens", 0) or 0),
                "completion_tokens": int(getattr(chunk.delta.usage.tokens, "output_tokens", 0) or 0),
                "total_tokens": int(
                    (getattr(chunk.delta.usage.tokens, "input_tokens", 0) or 0)
                    + (getattr(chunk.delta.usage.tokens, "output_tokens", 0) or 0)
                ),
            }

    choice_dict = {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
        "logprobs": None,
    }

    chunk_dict["choices"] = [choice_dict]

    return ChatCompletionChunk.model_validate(chunk_dict)


def _convert_response(response: V2ChatResponse, model: str) -> ChatCompletion:
    """Convert Cohere response to OpenAI ChatCompletion format directly."""
    prompt_tokens = 0
    completion_tokens = 0

    if response.usage and response.usage.tokens:
        prompt_tokens = int(response.usage.tokens.input_tokens or 0)
        completion_tokens = int(response.usage.tokens.output_tokens or 0)

    usage = CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    if response.finish_reason == "TOOL_CALL" and response.message.tool_calls:
        message = ChatCompletionMessage(
            role="assistant",
            content=response.message.tool_plan,
            tool_calls=[
                ChatCompletionMessageFunctionToolCall(
                    id=tc.id or "",
                    type="function",
                    function=Function(
                        name=tc.function.name if tc.function and tc.function.name else "",
                        arguments=tc.function.arguments if tc.function and tc.function.arguments else "",
                    ),
                )
                for tc in response.message.tool_calls
            ],
        )
        choice = Choice(index=0, finish_reason="tool_calls", message=message)
        return ChatCompletion(
            id=getattr(response, "id", ""),
            model=model,
            created=getattr(response, "created", 0),
            object="chat.completion",
            choices=[choice],
            usage=usage,
        )
    content = ""
    reasoning_content = None

    if response.message.content and len(response.message.content) > 0:
        for item in response.message.content:
            if hasattr(item, "type"):
                if item.type == "text" and hasattr(item, "text"):
                    content += item.text
                elif item.type == "thinking" and hasattr(item, "thinking"):
                    reasoning_content = Reasoning(content=item.thinking)

    message = ChatCompletionMessage(role="assistant", content=content, tool_calls=None, reasoning=reasoning_content)
    choice = Choice(
        index=0,
        finish_reason="stop",
        message=message,
    )
    return ChatCompletion(
        id=getattr(response, "id", ""),
        model=model,
        created=getattr(response, "created", 0),
        object="chat.completion",
        choices=[choice],
        usage=usage,
    )


def _convert_cohere_rerank_response(response: Any) -> RerankResponse:
    """Convert a Cohere V2 rerank response to a normalized RerankResponse."""
    results = [
        RerankResult(
            index=r.index,
            relevance_score=r.relevance_score,
        )
        for r in response.results
    ]
    # Defensive: Cohere returns sorted but re-sort to guarantee the docstring contract
    results.sort(key=lambda r: r.relevance_score, reverse=True)

    meta = None
    if hasattr(response, "meta") and response.meta is not None:
        billed_units = None
        tokens = None
        if hasattr(response.meta, "billed_units") and response.meta.billed_units is not None:
            billed_units = {}
            if (
                hasattr(response.meta.billed_units, "search_units")
                and response.meta.billed_units.search_units is not None
            ):
                billed_units["search_units"] = float(response.meta.billed_units.search_units)
        if hasattr(response.meta, "tokens") and response.meta.tokens is not None:
            tokens = {}
            if hasattr(response.meta.tokens, "input_tokens") and response.meta.tokens.input_tokens is not None:
                tokens["input_tokens"] = int(response.meta.tokens.input_tokens)
        if billed_units or tokens:
            meta = RerankMeta(billed_units=billed_units or None, tokens=tokens or None)

    usage = None
    if meta and meta.tokens and "input_tokens" in meta.tokens:
        usage = RerankUsage(total_tokens=meta.tokens["input_tokens"])

    return RerankResponse(
        id=response.id,
        results=results,
        meta=meta,
        usage=usage,
    )


_EMBEDDING_TYPE_FIELDS = ("float_", "int8", "uint8", "binary", "ubinary")


def _extract_vectors(embeddings_data: Any) -> list[list[float]]:
    """Return the first non-empty embedding vectors from a Cohere EmbedByTypeResponseEmbeddings.

    Integer-typed fields (int8, uint8, binary, ubinary) are cast to float.
    """
    for field in _EMBEDDING_TYPE_FIELDS:
        vectors = getattr(embeddings_data, field, None)
        if vectors:
            return [[float(v) for v in vec] for vec in vectors]
    return []


def _convert_cohere_embedding_response(model: str, response: Any) -> CreateEmbeddingResponse:
    """Convert a Cohere EmbedByTypeResponse to an OpenAI CreateEmbeddingResponse."""
    vectors = _extract_vectors(response.embeddings)

    openai_embeddings = [Embedding(embedding=vector, index=i, object="embedding") for i, vector in enumerate(vectors)]

    prompt_tokens = 0
    total_tokens = 0
    if response.meta:
        # Cohere reports embed usage under meta.billed_units.input_tokens; meta.tokens is
        # typically null for embeddings, so fall back to billed_units to avoid reporting 0.
        # Use explicit None checks so an API-reported 0 is preserved rather than treated as absent.
        if response.meta.tokens is not None and response.meta.tokens.input_tokens is not None:
            prompt_tokens = int(response.meta.tokens.input_tokens)
        elif response.meta.billed_units is not None and response.meta.billed_units.input_tokens is not None:
            prompt_tokens = int(response.meta.billed_units.input_tokens)
        total_tokens = prompt_tokens

    return CreateEmbeddingResponse(
        data=openai_embeddings,
        model=model,
        object="list",
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens),
    )


def _convert_models_list(response: CohereListModelsResponse) -> Sequence[Model]:
    """Converts a Cohere ListModelsResponse to a list of Model objects."""
    models = []
    if response.models:
        for model_data in response.models:
            models.append(
                Model(
                    id=model_data.name or "unknown",
                    created=0,  # Cohere doesn't provide this, so we use a default value
                    object="model",
                    owned_by="cohere",
                )
            )
    return models
