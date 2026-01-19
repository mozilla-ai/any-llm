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

from any_llm.logging import logger
from any_llm.types.batch import Batch, BatchRequestCounts
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

DEFAULT_TIMEOUT_HOURS = 24
DEFAULT_COMPLETION_WINDOW = f"{DEFAULT_TIMEOUT_HOURS}h"

if TYPE_CHECKING:
    from mistralai.models import BatchJobOut, BatchJobsOut
    from mistralai.models.embeddingresponse import EmbeddingResponse
    from openai.types.chat.chat_completion_message_custom_tool_call import (
        ChatCompletionMessageCustomToolCall,
    )
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall as OpenAIChatCompletionMessageFunctionToolCall,
    )

    ChatCompletionMessageToolCallType = (
        OpenAIChatCompletionMessageFunctionToolCall | ChatCompletionMessageCustomToolCall
    )


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
            tool_calls=cast("list[ChatCompletionMessageToolCallType] | None", tool_calls_final),
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


_MISTRAL_TO_OPENAI_STATUS_MAP: dict[str, str] = {
    "QUEUED": "validating",
    "RUNNING": "in_progress",
    "SUCCESS": "completed",
    "FAILED": "failed",
    "TIMEOUT_EXCEEDED": "expired",
    "CANCELLATION_REQUESTED": "cancelling",
    "CANCELLED": "cancelled",
}


def _convert_batch_job_to_openai(batch_job: "BatchJobOut") -> Batch:
    """Convert a Mistral BatchJobOut to OpenAI Batch format."""
    status = batch_job.status
    status_str = str(status.value if hasattr(status, "value") else status)  # type: ignore[union-attr]
    openai_status = _MISTRAL_TO_OPENAI_STATUS_MAP.get(status_str)
    if openai_status is None:
        logger.warning(f"Unknown Mistral batch status: {status_str}, defaulting to 'in_progress'")
        openai_status = "in_progress"

    request_counts = BatchRequestCounts(
        total=batch_job.total_requests,
        completed=batch_job.completed_requests,
        failed=batch_job.failed_requests,
    )

    input_file_id = batch_job.input_files[0] if batch_job.input_files else ""

    metadata: dict[str, str] | None = None
    if batch_job.metadata is not None and not isinstance(batch_job.metadata, Unset):
        metadata = {k: str(v) for k, v in batch_job.metadata.items()}

    output_file_id: str | None = None
    if batch_job.output_file is not None and not isinstance(batch_job.output_file, Unset):
        output_file_id = batch_job.output_file

    error_file_id: str | None = None
    if batch_job.error_file is not None and not isinstance(batch_job.error_file, Unset):
        error_file_id = batch_job.error_file

    started_at: int | None = None
    if batch_job.started_at is not None and not isinstance(batch_job.started_at, Unset):
        started_at = batch_job.started_at

    completed_at: int | None = None
    if batch_job.completed_at is not None and not isinstance(batch_job.completed_at, Unset):
        completed_at = batch_job.completed_at

    completion_window = DEFAULT_COMPLETION_WINDOW
    if hasattr(batch_job, "timeout_hours") and batch_job.timeout_hours is not None:
        if not isinstance(batch_job.timeout_hours, Unset):
            completion_window = f"{batch_job.timeout_hours}h"

    return Batch(
        id=batch_job.id,
        object="batch",
        endpoint=batch_job.endpoint,
        input_file_id=input_file_id,
        completion_window=completion_window,
        status=cast(
            "Literal['validating', 'failed', 'in_progress', 'finalizing', 'completed', 'expired', 'cancelling', 'cancelled']",
            openai_status,
        ),
        created_at=batch_job.created_at,
        in_progress_at=started_at,
        completed_at=completed_at,
        output_file_id=output_file_id,
        error_file_id=error_file_id,
        request_counts=request_counts,
        metadata=metadata,
    )


def _convert_batch_jobs_list(batch_jobs: "BatchJobsOut") -> Sequence[Batch]:
    """Convert a Mistral BatchJobsOut to a sequence of OpenAI Batch objects."""
    if batch_jobs.data is None:
        return []
    return [_convert_batch_job_to_openai(job) for job in batch_jobs.data]


class MixedModelError(ValueError):
    """Raised when a batch file contains requests with different model IDs."""

    def __init__(self, models_found: set[str]) -> None:
        self.models_found = models_found
        super().__init__(
            f"Mistral batch API requires all requests to use the same model. "
            f"Found {len(models_found)} different models: {sorted(models_found)}"
        )


def _parse_completion_window_to_hours(completion_window: str) -> int:
    """
    Convert OpenAI-style completion_window string to Mistral timeout_hours integer.

    OpenAI currently only supports "24h" as the completion_window value.
    This function parses that format (e.g., "24h", "48h") and returns the integer hours.

    Args:
        completion_window: OpenAI-style completion window string (e.g., "24h").

    Returns:
        Integer number of hours for Mistral's timeout_hours parameter.

    Raises:
        ValueError: If the format is not recognized.
    """
    window = completion_window.strip().lower()

    if not window:
        return DEFAULT_TIMEOUT_HOURS

    if not window.endswith("h"):
        msg = f"Invalid completion_window format: '{completion_window}'. Expected format like '24h'."
        raise ValueError(msg)

    try:
        hours = int(window[:-1])
    except ValueError:
        msg = f"Invalid completion_window format: '{completion_window}'. Expected format like '24h'."
        raise ValueError(msg) from None

    if hours <= 0:
        msg = f"completion_window must be positive, got: '{completion_window}'"
        raise ValueError(msg)

    return hours


def _validate_batch_file_models(file_content: str) -> str | None:
    """
    Validate that all requests in a JSONL batch file use the same model.

    Mistral's batch API requires specifying the model at the job level, not per-request.
    This function ensures all requests target the same model and returns that model ID.

    Args:
        file_content: The content of the JSONL batch file as a string.

    Returns:
        The model ID used across all requests, or None if no models are specified.

    Raises:
        MixedModelError: If different models are found in the batch file.
        ValueError: If the file is empty or contains invalid JSON.
    """
    if not file_content.strip():
        msg = "Input file is empty"
        raise ValueError(msg)

    lines = [line.strip() for line in file_content.strip().split("\n") if line.strip()]

    models_found: set[str] = set()

    for line_num, line in enumerate(lines, start=1):
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            line_desc = "first line" if line_num == 1 else f"line {line_num}"
            msg = f"Invalid JSONL format in {line_desc}: {e}"
            raise ValueError(msg) from e

        body = request.get("body", {})
        model = body.get("model")

        if model:
            models_found.add(model)

    if len(models_found) > 1:
        raise MixedModelError(models_found)

    return models_found.pop() if models_found else None
