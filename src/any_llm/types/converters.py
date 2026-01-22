"""Converters between OpenAI SDK types and OpenResponses spec types.

This module provides conversion functions to translate between OpenAI SDK
response types and the OpenResponses specification types from the
openresponses-types package.

This allows the library to use pure OpenResponses types in its public API
while still using the OpenAI SDK internally for provider implementations.

Note: This module uses dynamic type conversions between two different type systems
(OpenAI SDK and OpenResponses spec). Type checking is disabled at the module level
because the generated OpenResponses types use strict enum types (Type31, Type38, etc.)
that are difficult to match statically with OpenAI SDK string literals.
"""
# mypy: ignore-errors

from typing import Any

from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseStreamEvent as OpenAIResponseStreamEvent
from openresponses_types.types import (
    Error,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    FunctionToolChoice,
    IncompleteDetails,
    InputTokensDetails,
    Message,
    MessageRole,
    MessageStatus,
    Object,
    OutputTextContent,
    OutputTokensDetails,
    Reasoning,
    ReasoningBody,
    ResponseCompletedStreamingEvent,
    ResponseContentPartAddedStreamingEvent,
    ResponseContentPartDoneStreamingEvent,
    ResponseCreatedStreamingEvent,
    ResponseFailedStreamingEvent,
    ResponseFunctionCallArgumentsDeltaStreamingEvent,
    ResponseFunctionCallArgumentsDoneStreamingEvent,
    ResponseIncompleteStreamingEvent,
    ResponseInProgressStreamingEvent,
    ResponseOutputItemAddedStreamingEvent,
    ResponseOutputItemDoneStreamingEvent,
    ResponseOutputTextAnnotationAddedStreamingEvent,
    ResponseOutputTextDeltaStreamingEvent,
    ResponseOutputTextDoneStreamingEvent,
    ResponseQueuedStreamingEvent,
    ResponseReasoningDeltaStreamingEvent,
    ResponseReasoningDoneStreamingEvent,
    ResponseReasoningSummaryDeltaStreamingEvent,
    ResponseReasoningSummaryDoneStreamingEvent,
    ResponseReasoningSummaryPartAddedStreamingEvent,
    ResponseReasoningSummaryPartDoneStreamingEvent,
    ResponseRefusalDeltaStreamingEvent,
    ResponseRefusalDoneStreamingEvent,
    ResponseResource,
    TextField,
    TextResponseFormat,
    Tool,
    ToolChoiceValueEnum,
    TruncationEnum,
    Type34,
    Type37,
    Type38,
    Type39,
    Type40,
    Type41,
    Type42,
    Usage,
)


def convert_openai_response_to_openresponses(response: OpenAIResponse) -> ResponseResource:
    """Convert an OpenAI SDK Response to an OpenResponses ResponseResource.

    Args:
        response: The OpenAI SDK Response object.

    Returns:
        An OpenResponses ResponseResource with equivalent data.
    """
    output_items: list[Message | FunctionCall | FunctionCallOutput | ReasoningBody] = []
    for item in response.output:
        output_items.append(_convert_output_item(item))

    tools: list[Tool] = []
    for tool in response.tools:
        tools.append(_convert_tool(tool))

    tool_choice = _convert_tool_choice(response.tool_choice)

    incomplete_details = None
    if response.incomplete_details:
        incomplete_details = IncompleteDetails(reason=response.incomplete_details.reason)

    error = None
    if response.error:
        error = Error(
            code=response.error.code or "",
            message=response.error.message or "",
        )

    reasoning = None
    if response.reasoning:
        reasoning = Reasoning(
            effort=response.reasoning.effort,
            summary=response.reasoning.summary,
        )

    usage = None
    if response.usage:
        input_details = InputTokensDetails(cached_tokens=0)
        output_details = OutputTokensDetails(reasoning_tokens=0)
        if hasattr(response.usage, "input_tokens_details") and response.usage.input_tokens_details:
            input_details = InputTokensDetails(
                cached_tokens=getattr(response.usage.input_tokens_details, "cached_tokens", 0)
            )
        if hasattr(response.usage, "output_tokens_details") and response.usage.output_tokens_details:
            output_details = OutputTokensDetails(
                reasoning_tokens=getattr(response.usage.output_tokens_details, "reasoning_tokens", 0)
            )
        usage = Usage(
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
            total_tokens=response.usage.total_tokens or 0,
            input_tokens_details=input_details,
            output_tokens_details=output_details,
        )

    text = TextField(format=TextResponseFormat(type=Type34.text))
    if response.text:
        text = TextField(format=TextResponseFormat(type=Type34.text))

    return ResponseResource(
        id=response.id,
        object=Object.response,
        created_at=int(response.created_at),
        completed_at=int(response.completed_at) if response.completed_at else None,
        status=response.status or "completed",
        incomplete_details=incomplete_details,
        model=str(response.model),
        previous_response_id=response.previous_response_id,
        instructions=response.instructions if isinstance(response.instructions, str) else None,
        output=output_items,
        error=error,
        tools=tools,
        tool_choice=tool_choice,
        truncation=TruncationEnum(response.truncation) if response.truncation else TruncationEnum.disabled,
        parallel_tool_calls=response.parallel_tool_calls if response.parallel_tool_calls is not None else True,
        text=text,
        top_p=response.top_p or 1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        top_logprobs=response.top_logprobs or 0,
        temperature=response.temperature or 1.0,
        reasoning=reasoning,
        usage=usage,
        max_output_tokens=response.max_output_tokens,
        max_tool_calls=response.max_tool_calls,
        store=False,
        background=response.background or False,
        service_tier=response.service_tier or "default",
        metadata=response.metadata,
        safety_identifier=response.safety_identifier,
        prompt_cache_key=response.prompt_cache_key,
    )


def _convert_output_item(item: Any) -> Message | FunctionCall | FunctionCallOutput | ReasoningBody:
    """Convert an OpenAI output item to OpenResponses format."""
    item_type = getattr(item, "type", None)

    if item_type == "message":
        content_list = []
        for content in getattr(item, "content", []):
            content_type = getattr(content, "type", None)
            if content_type == "output_text":
                content_list.append(
                    OutputTextContent(
                        type="output_text",
                        text=getattr(content, "text", ""),
                        annotations=[],
                        logprobs=[],
                    )
                )
        return Message(
            type="message",
            id=getattr(item, "id", ""),
            status=MessageStatus(getattr(item, "status", "completed")),
            role=MessageRole(getattr(item, "role", "assistant")),
            content=content_list,
        )
    if item_type == "function_call":
        return FunctionCall(
            type="function_call",
            id=getattr(item, "id", ""),
            call_id=getattr(item, "call_id", ""),
            name=getattr(item, "name", ""),
            arguments=getattr(item, "arguments", ""),
            status=getattr(item, "status", "completed"),
        )
    if item_type == "function_call_output":
        return FunctionCallOutput(
            type="function_call_output",
            id=getattr(item, "id", ""),
            call_id=getattr(item, "call_id", ""),
            output=getattr(item, "output", ""),
        )
    if item_type == "reasoning":
        return ReasoningBody(
            type="reasoning",
            id=getattr(item, "id", ""),
            status=getattr(item, "status", "completed"),
            summary=[],
        )
    return Message(
        type="message",
        id=getattr(item, "id", ""),
        status=MessageStatus.completed,
        role=MessageRole.assistant,
        content=[],
    )


def _convert_tool(tool: Any) -> Tool:
    """Convert an OpenAI tool to OpenResponses format."""
    tool_type = getattr(tool, "type", None)

    if tool_type == "function":
        return Tool(
            root=FunctionTool(
                type="function",
                name=getattr(tool, "name", ""),
                description=getattr(tool, "description", None),
                parameters=getattr(tool, "parameters", None),
                strict=getattr(tool, "strict", None),
            )
        )
    return Tool(
        root=FunctionTool(
            type="function",
            name="unknown",
            description=None,
            parameters=None,
            strict=None,
        )
    )


def _convert_tool_choice(tool_choice: Any) -> FunctionToolChoice | ToolChoiceValueEnum:
    """Convert OpenAI tool choice to OpenResponses format."""
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "none", "required"):
            return ToolChoiceValueEnum(tool_choice)
    if hasattr(tool_choice, "type") and tool_choice.type == "function":
        return FunctionToolChoice(
            type="function",
            name=getattr(tool_choice, "name", ""),
        )
    return ToolChoiceValueEnum.auto


# Mapping from OpenAI event types to OpenResponses event types
_EVENT_TYPE_MAP = {
    "response.created": Type37.response_created,
    "response.queued": Type38.response_queued,
    "response.in_progress": Type39.response_in_progress,
    "response.completed": Type40.response_completed,
    "response.failed": Type41.response_failed,
    "response.incomplete": Type42.response_incomplete,
}


def convert_openai_stream_event(event: OpenAIResponseStreamEvent) -> Any:
    """Convert an OpenAI SDK streaming event to OpenResponses format.

    Args:
        event: The OpenAI SDK streaming event.

    Returns:
        An OpenResponses streaming event with equivalent data.

    Note:
        Some OpenAI-specific events (web search, code interpreter, etc.) may
        not have direct OpenResponses equivalents and will be passed through
        with minimal conversion.
    """
    event_type = getattr(event, "type", "unknown")
    sequence_number = getattr(event, "sequence_number", 0)

    if event_type == "response.created":
        return ResponseCreatedStreamingEvent(
            type=Type37.response_created,
            sequence_number=sequence_number,
            response=_convert_response_for_event(event),
        )
    if event_type == "response.queued":
        return ResponseQueuedStreamingEvent(
            type=Type38.response_queued,
            sequence_number=sequence_number,
            response=_convert_response_for_event(event),
        )
    if event_type == "response.in_progress":
        return ResponseInProgressStreamingEvent(
            type=Type39.response_in_progress,
            sequence_number=sequence_number,
            response=_convert_response_for_event(event),
        )
    if event_type == "response.completed":
        return ResponseCompletedStreamingEvent(
            type=Type40.response_completed,
            sequence_number=sequence_number,
            response=_convert_response_for_event(event),
        )
    if event_type == "response.failed":
        return ResponseFailedStreamingEvent(
            type=Type41.response_failed,
            sequence_number=sequence_number,
            response=_convert_response_for_event(event),
        )
    if event_type == "response.incomplete":
        return ResponseIncompleteStreamingEvent(
            type=Type42.response_incomplete,
            sequence_number=sequence_number,
            response=_convert_response_for_event(event),
        )
    if event_type == "response.output_item.added":
        return ResponseOutputItemAddedStreamingEvent(
            type="response.output_item.added",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            item=_convert_output_item(getattr(event, "item", None)),
        )
    if event_type == "response.output_item.done":
        return ResponseOutputItemDoneStreamingEvent(
            type="response.output_item.done",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            item=_convert_output_item(getattr(event, "item", None)),
        )
    if event_type == "response.output_text.delta":
        return ResponseOutputTextDeltaStreamingEvent(
            type="response.output_text.delta",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            content_index=getattr(event, "content_index", 0),
            delta=getattr(event, "delta", ""),
        )
    if event_type == "response.output_text.done":
        return ResponseOutputTextDoneStreamingEvent(
            type="response.output_text.done",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            content_index=getattr(event, "content_index", 0),
            text=getattr(event, "text", ""),
        )
    if event_type == "response.content_part.added":
        return ResponseContentPartAddedStreamingEvent(
            type="response.content_part.added",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            content_index=getattr(event, "content_index", 0),
            part=_convert_content_part(getattr(event, "part", None)),
        )
    if event_type == "response.content_part.done":
        return ResponseContentPartDoneStreamingEvent(
            type="response.content_part.done",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            content_index=getattr(event, "content_index", 0),
            part=_convert_content_part(getattr(event, "part", None)),
        )
    if event_type == "response.refusal.delta":
        return ResponseRefusalDeltaStreamingEvent(
            type="response.refusal.delta",
            sequence_number=sequence_number,
            delta=getattr(event, "delta", ""),
        )
    if event_type == "response.refusal.done":
        return ResponseRefusalDoneStreamingEvent(
            type="response.refusal.done",
            sequence_number=sequence_number,
            refusal=getattr(event, "refusal", ""),
        )
    if event_type == "response.function_call_arguments.delta":
        return ResponseFunctionCallArgumentsDeltaStreamingEvent(
            type="response.function_call_arguments.delta",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            delta=getattr(event, "delta", ""),
        )
    if event_type == "response.function_call_arguments.done":
        return ResponseFunctionCallArgumentsDoneStreamingEvent(
            type="response.function_call_arguments.done",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            arguments=getattr(event, "arguments", ""),
        )
    if event_type == "response.reasoning_summary.text.delta":
        return ResponseReasoningSummaryDeltaStreamingEvent(
            type="response.reasoning_summary_text.delta",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            summary_index=getattr(event, "summary_index", 0),
            delta=getattr(event, "delta", ""),
        )
    if event_type == "response.reasoning_summary.text.done":
        return ResponseReasoningSummaryDoneStreamingEvent(
            type="response.reasoning_summary_text.done",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            summary_index=getattr(event, "summary_index", 0),
            text=getattr(event, "text", ""),
        )
    if event_type == "response.reasoning_summary.part.added":
        return ResponseReasoningSummaryPartAddedStreamingEvent(
            type="response.reasoning_summary.part.added",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            summary_index=getattr(event, "summary_index", 0),
            part=_convert_reasoning_summary_part(getattr(event, "part", None)),
        )
    if event_type == "response.reasoning_summary.part.done":
        return ResponseReasoningSummaryPartDoneStreamingEvent(
            type="response.reasoning_summary.part.done",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            summary_index=getattr(event, "summary_index", 0),
            part=_convert_reasoning_summary_part(getattr(event, "part", None)),
        )
    if event_type == "response.reasoning.delta":
        return ResponseReasoningDeltaStreamingEvent(
            type="response.reasoning.delta",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            delta=getattr(event, "delta", ""),
        )
    if event_type == "response.reasoning.done":
        return ResponseReasoningDoneStreamingEvent(
            type="response.reasoning.done",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            text=getattr(event, "text", ""),
        )
    if event_type == "response.output_text.annotation.added":
        return ResponseOutputTextAnnotationAddedStreamingEvent(
            type="response.output_text.annotation.added",
            sequence_number=sequence_number,
            output_index=getattr(event, "output_index", 0),
            content_index=getattr(event, "content_index", 0),
            annotation_index=getattr(event, "annotation_index", 0),
            annotation=_convert_annotation(getattr(event, "annotation", None)),
        )
    # For events without direct OpenResponses equivalents, return a generic event
    # This handles OpenAI-specific events like web_search, code_interpreter, etc.
    return _create_passthrough_event(event)


def _convert_response_for_event(event: Any) -> ResponseResource:
    """Convert the response object embedded in a streaming event."""
    response = getattr(event, "response", None)
    if response is None:
        return _create_minimal_response()

    if hasattr(response, "id"):
        return convert_openai_response_to_openresponses(response)

    return _create_minimal_response()


def _create_minimal_response() -> ResponseResource:
    """Create a minimal ResponseResource for events without full response data."""
    return ResponseResource(
        id="",
        object=Object.response,
        created_at=0,
        completed_at=None,
        status="in_progress",
        incomplete_details=None,
        model="",
        previous_response_id=None,
        instructions=None,
        output=[],
        error=None,
        tools=[],
        tool_choice=ToolChoiceValueEnum.auto,
        truncation=TruncationEnum.disabled,
        parallel_tool_calls=True,
        text=TextField(format=TextResponseFormat(type=Type34.text)),
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        top_logprobs=0,
        temperature=1.0,
        reasoning=None,
        usage=None,
        max_output_tokens=None,
        max_tool_calls=None,
        store=False,
        background=False,
        service_tier="default",
        metadata=None,
        safety_identifier=None,
        prompt_cache_key=None,
    )


def _convert_content_part(part: Any) -> OutputTextContent:
    """Convert a content part from OpenAI format to OpenResponses format."""
    if part is None:
        return OutputTextContent(type="output_text", text="", annotations=[], logprobs=[])

    part_type = getattr(part, "type", "output_text")
    if part_type == "output_text":
        return OutputTextContent(
            type="output_text",
            text=getattr(part, "text", ""),
            annotations=[],
            logprobs=[],
        )
    return OutputTextContent(type="output_text", text="", annotations=[], logprobs=[])


def _convert_reasoning_summary_part(part: Any) -> Any:
    """Convert a reasoning summary part."""
    if part is None:
        return {"type": "summary_text", "text": ""}
    return {
        "type": getattr(part, "type", "summary_text"),
        "text": getattr(part, "text", ""),
    }


def _convert_annotation(annotation: Any) -> Any:
    """Convert an annotation from OpenAI format."""
    if annotation is None:
        return {"type": "url_citation", "url": "", "title": ""}
    return {
        "type": getattr(annotation, "type", "url_citation"),
        "url": getattr(annotation, "url", ""),
        "title": getattr(annotation, "title", ""),
    }


def _create_passthrough_event(event: Any) -> Any:
    """Create a passthrough event for OpenAI-specific event types.

    Events like web_search, code_interpreter, image_gen, and mcp calls
    don't have direct OpenResponses equivalents, so we create a generic
    structure that preserves the event data.
    """
    return {
        "type": getattr(event, "type", "unknown"),
        "sequence_number": getattr(event, "sequence_number", 0),
        "data": {k: v for k, v in vars(event).items() if not k.startswith("_")},
    }
