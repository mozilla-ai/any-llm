import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from together.types import ChatCompletionChunk as TogetherChatCompletionChunk

from any_llm.logging import logger
from any_llm.types.batch import Batch
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    Reasoning,
    Usage,
)
from any_llm.types.model import Model
from any_llm.utils.reasoning import normalize_reasoning_from_provider_fields_and_xml_tags

if TYPE_CHECKING:
    from together.types import BatchJob as TogetherBatchJob
    from together.types import Embedding as TogetherEmbedding

DEFAULT_COMPLETION_WINDOW = "24h"

_TOGETHER_TO_OPENAI_BATCH_STATUS: dict[str, str] = {
    "VALIDATING": "validating",
    "IN_PROGRESS": "in_progress",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "EXPIRED": "expired",
    "CANCELING": "cancelling",
    "CANCELLING": "cancelling",
    "CANCELLED": "cancelled",
}


def _create_openai_chunk_from_together_chunk(together_chunk: TogetherChatCompletionChunk) -> ChatCompletionChunk:
    """Convert a Together streaming chunk to OpenAI ChatCompletionChunk format."""

    openai_choices: list[ChunkChoice] = []
    for choice in together_chunk.choices or []:
        delta_content = choice.delta
        content = None
        role = None
        reasoning = None

        if delta_content:
            if not hasattr(delta_content, "content"):
                logger.warning("Together delta_content missing 'content' attribute: %s", delta_content)
            content = getattr(delta_content, "content", None)
            if getattr(delta_content, "role", None):
                role = cast("Literal['assistant', 'user', 'system']", delta_content.role)
            if hasattr(delta_content, "reasoning") and delta_content.reasoning:
                reasoning = Reasoning(content=delta_content.reasoning)

        delta = ChoiceDelta(content=content, role=role, reasoning=reasoning)

        if delta_content and hasattr(delta_content, "tool_calls") and delta_content.tool_calls:
            openai_tool_calls = []
            for idx, tool_call in enumerate(delta_content.tool_calls):
                if isinstance(tool_call, dict):
                    func = tool_call.get("function", {})
                    tc_id = tool_call.get("id") or str(uuid.uuid4())
                    raw_index = tool_call.get("index")
                    tc_index = raw_index if raw_index is not None else idx
                    name = func.get("name", "")
                    arguments = func.get("arguments", "")
                else:
                    tc_id = getattr(tool_call, "id", None) or str(uuid.uuid4())
                    raw_index = getattr(tool_call, "index", None)
                    tc_index = raw_index if raw_index is not None else idx
                    func = getattr(tool_call, "function", None)
                    name = getattr(func, "name", "") if func else ""
                    arguments = getattr(func, "arguments", "") if func else ""

                openai_tool_call = ChoiceDeltaToolCall(
                    index=tc_index,
                    id=tc_id,
                    type="function",
                    function=ChoiceDeltaToolCallFunction(
                        name=name,
                        arguments=arguments,
                    ),
                )
                openai_tool_calls.append(openai_tool_call)
            delta.tool_calls = openai_tool_calls

        openai_choice = ChunkChoice(
            index=choice.index or len(openai_choices),
            delta=delta,
            finish_reason=cast(
                "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'] | None",
                choice.finish_reason,
            ),
        )
        openai_choices.append(openai_choice)

    usage = None
    if together_chunk.usage:
        usage = CompletionUsage(
            prompt_tokens=together_chunk.usage.prompt_tokens or 0,
            completion_tokens=together_chunk.usage.completion_tokens or 0,
            total_tokens=together_chunk.usage.total_tokens or 0,
        )

    return ChatCompletionChunk(
        id=together_chunk.id or f"chatcmpl-{uuid.uuid4()}",
        choices=openai_choices,
        created=together_chunk.created or int(datetime.now().timestamp()),
        model=together_chunk.model or "unknown",
        object="chat.completion.chunk",
        usage=usage,
    )


def _convert_together_response_to_chat_completion(response_data: dict[str, Any], model_id: str) -> ChatCompletion:
    """Convert Together API response to OpenAI ChatCompletion format."""
    choices_out: list[Choice] = []
    for i, ch in enumerate(response_data.get("choices", [])):
        msg = ch.get("message", {})

        normalize_reasoning_from_provider_fields_and_xml_tags(msg)

        message = ChatCompletionMessage(
            role=cast("Literal['assistant']", msg.get("role")),
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls"),
            reasoning=msg.get("reasoning"),
        )
        choices_out.append(
            Choice(
                index=i,
                finish_reason=cast(
                    "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']",
                    ch.get("finish_reason"),
                ),
                message=message,
            )
        )

    usage = None
    if response_data.get("usage"):
        u = response_data["usage"]
        usage = CompletionUsage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )

    return ChatCompletion(
        id=response_data.get("id", ""),
        model=model_id,
        created=response_data.get("created", 0),
        object="chat.completion",
        choices=choices_out,
        usage=usage,
    )


def _convert_models_list(response: Any) -> list[Model]:
    """Convert Together model listing response to OpenAI-compatible Model objects."""
    raw_models = response.data if hasattr(response, "data") else response
    converted_models: list[Model] = []

    for model in raw_models:
        data = model.model_dump() if hasattr(model, "model_dump") else dict(vars(model))
        if data.get("object") is None:
            data["object"] = "model"
        if data.get("created") is None:
            data["created"] = 0
        if data.get("owned_by") is None:
            data["owned_by"] = data.get("organization") or "together"
        converted_models.append(Model.model_validate(data))

    return converted_models


def _create_openai_embedding_response_from_together(
    response: "TogetherEmbedding",
) -> CreateEmbeddingResponse:
    """Convert a Together Embedding response to OpenAI CreateEmbeddingResponse format."""
    embeddings = [
        Embedding(embedding=list(entry.embedding), index=entry.index, object="embedding") for entry in response.data
    ]

    # Together returns token usage but the SDK model does not declare the field, so it
    # arrives as a pydantic extra rather than a typed attribute.
    raw_usage: Any = getattr(response, "usage", None)
    if hasattr(raw_usage, "model_dump"):
        raw_usage = raw_usage.model_dump()
    usage_data = raw_usage if isinstance(raw_usage, dict) else {}
    usage = Usage(
        prompt_tokens=usage_data.get("prompt_tokens") or 0,
        total_tokens=usage_data.get("total_tokens") or 0,
    )

    return CreateEmbeddingResponse(
        data=embeddings,
        model=response.model,
        object="list",
        usage=usage,
    )


def _datetime_to_epoch(value: datetime | None) -> int | None:
    """Convert a Together timestamp to the integer epoch seconds OpenAI's Batch uses."""
    if value is None:
        return None
    return int(value.timestamp())


def _derive_completion_window(batch_job: "TogetherBatchJob") -> str:
    """Recover the completion window from the job deadline."""
    if batch_job.created_at is None or batch_job.job_deadline is None:
        return DEFAULT_COMPLETION_WINDOW
    hours = round((batch_job.job_deadline - batch_job.created_at).total_seconds() / 3600)
    if hours <= 0:
        return DEFAULT_COMPLETION_WINDOW
    return f"{hours}h"


def _convert_batch_job_to_openai(batch_job: "TogetherBatchJob") -> Batch:
    """Convert a Together BatchJob to OpenAI Batch format."""
    status = _TOGETHER_TO_OPENAI_BATCH_STATUS.get(str(batch_job.status).upper())
    if status is None:
        logger.warning("Unknown Together batch status: %s, defaulting to 'in_progress'", batch_job.status)
        status = "in_progress"

    return Batch(
        id=batch_job.id or "",
        object="batch",
        endpoint=batch_job.endpoint or "",
        input_file_id=batch_job.input_file_id or "",
        completion_window=_derive_completion_window(batch_job),
        status=cast(
            "Literal['validating', 'failed', 'in_progress', 'finalizing', 'completed', 'expired', 'cancelling', 'cancelled']",
            status,
        ),
        created_at=_datetime_to_epoch(batch_job.created_at) or 0,
        completed_at=_datetime_to_epoch(batch_job.completed_at),
        expires_at=_datetime_to_epoch(batch_job.job_deadline),
        output_file_id=batch_job.output_file_id,
        error_file_id=batch_job.error_file_id,
        model=batch_job.x_model_id,
    )
