from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Literal, cast

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    Reasoning,
    Usage,
)
from any_llm.types.model import Model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lmstudio import (
        AnyAsyncDownloadedModel,
        LlmPredictionFragment,
        LlmPredictionStats,
        PredictionResult,
    )

OpenAIFinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]

# Map LM Studio stop reasons to the OpenAI-style finish reasons any-llm exposes.
# https://lmstudio.ai/docs/python (LlmPredictionStopReason)
_STOP_REASON_MAP: dict[str, OpenAIFinishReason] = {
    "eosFound": "stop",
    "stopStringFound": "stop",
    "userStopped": "stop",
    "modelUnloaded": "stop",
    "failed": "stop",
    "toolCalls": "tool_calls",
    "maxPredictedTokensReached": "length",
    "contextLengthReached": "length",
}


def _map_stop_reason(stop_reason: str | None) -> OpenAIFinishReason:
    if stop_reason is None:
        return "stop"
    return _STOP_REASON_MAP.get(stop_reason, "stop")


def _usage_from_stats(stats: LlmPredictionStats | None) -> CompletionUsage:
    prompt_tokens = int(stats.prompt_tokens_count or 0) if stats else 0
    completion_tokens = int(stats.predicted_tokens_count or 0) if stats else 0
    total_tokens = int(stats.total_tokens_count or 0) if stats else 0
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _split_reasoning_from_content(content: str) -> tuple[str, str | None]:
    """Extract reasoning wrapped in <think></think> tags from the content (if present).

    Any text before the opening tag or after the closing tag is preserved as content,
    so a model that emits a preface before its reasoning block does not lose output.
    """
    if "<think>" in content and "</think>" in content:
        before, after_open = content.split("<think>", 1)
        reasoning, after_close = after_open.split("</think>", 1)
        return before + after_close, reasoning
    return content, None


def create_completion_from_prediction(result: PredictionResult, model: str) -> ChatCompletion:
    """Convert an LM Studio PredictionResult into an OpenAI-compatible ChatCompletion."""
    content, reasoning_content = _split_reasoning_from_content(result.content)

    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        reasoning=Reasoning(content=reasoning_content) if reasoning_content else None,
    )
    choice = Choice(
        index=0,
        finish_reason=_map_stop_reason(result.stats.stop_reason if result.stats else None),
        message=message,
    )
    return ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4()}",
        model=model,
        created=int(time.time()),
        object="chat.completion",
        choices=[choice],
        usage=_usage_from_stats(result.stats),
    )


def create_chunk_from_fragment(
    fragment: LlmPredictionFragment,
    model: str,
    response_id: str,
) -> ChatCompletionChunk:
    """Convert a streaming LM Studio prediction fragment into an OpenAI-compatible chunk."""
    if fragment.reasoning_type == "reasoning":
        delta = ChoiceDelta(role="assistant", reasoning=Reasoning(content=fragment.content))
    elif fragment.reasoning_type in ("reasoningStartTag", "reasoningEndTag"):
        # The tag delimiters carry no user-facing content of their own.
        delta = ChoiceDelta(role="assistant")
    else:
        delta = ChoiceDelta(role="assistant", content=fragment.content)

    return ChatCompletionChunk(
        id=response_id,
        choices=[ChunkChoice(index=0, delta=delta, finish_reason=None)],
        created=int(time.time()),
        model=model,
        object="chat.completion.chunk",
    )


def create_final_chunk(result: PredictionResult, model: str, response_id: str) -> ChatCompletionChunk:
    """Emit a terminal chunk carrying the finish reason and usage statistics."""
    return ChatCompletionChunk(
        id=response_id,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason=_map_stop_reason(result.stats.stop_reason if result.stats else None),
            )
        ],
        created=int(time.time()),
        model=model,
        object="chat.completion.chunk",
        usage=_usage_from_stats(result.stats),
    )


def create_embedding_response(model: str, vectors: list[list[float]]) -> CreateEmbeddingResponse:
    """Convert LM Studio embedding vectors into an OpenAI-compatible CreateEmbeddingResponse."""
    data = [Embedding(embedding=vector, index=index, object="embedding") for index, vector in enumerate(vectors)]
    return CreateEmbeddingResponse(
        data=data,
        model=model,
        object="list",
        usage=Usage(prompt_tokens=0, total_tokens=0),
    )


def convert_models_list(downloaded_models: Sequence[AnyAsyncDownloadedModel]) -> list[Model]:
    """Convert LM Studio downloaded-model listings into OpenAI-compatible Model entries."""
    return [
        Model(
            id=cast("Any", model).model_key,
            object="model",
            created=0,
            owned_by="lmstudio",
        )
        for model in downloaded_models
    ]
