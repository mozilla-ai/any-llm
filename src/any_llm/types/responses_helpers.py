"""Helper types and aliases for OpenResponses API.

This module provides:
- Type aliases mapping common names to generated types
- Custom types not yet in the OpenResponses spec (e.g., MCP tools)
- Helper functions for working with Response objects

Import from here for convenience, or import directly from responses.py
for the raw generated types.
"""

from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from any_llm.types.responses import (
    AssistantMessageItemParam,
    CreateResponseBody,
    DeveloperMessageItemParam,
    FunctionCallItemParam,
    FunctionCallOutputItemParam,
    FunctionToolParam,
    Input,
    ItemReferenceParam,
    ReasoningEffortEnum,
    ReasoningItemParam,
    ReasoningParam,
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
    SystemMessageItemParam,
    UserMessageItemParam,
)

# =============================================================================
# Type Aliases - map common names to generated types
# =============================================================================

Response = ResponseResource

ResponseStreamEvent = (
    ResponseCreatedStreamingEvent
    | ResponseQueuedStreamingEvent
    | ResponseInProgressStreamingEvent
    | ResponseCompletedStreamingEvent
    | ResponseFailedStreamingEvent
    | ResponseIncompleteStreamingEvent
    | ResponseOutputItemAddedStreamingEvent
    | ResponseOutputItemDoneStreamingEvent
    | ResponseContentPartAddedStreamingEvent
    | ResponseContentPartDoneStreamingEvent
    | ResponseOutputTextDeltaStreamingEvent
    | ResponseOutputTextDoneStreamingEvent
    | ResponseRefusalDeltaStreamingEvent
    | ResponseRefusalDoneStreamingEvent
    | ResponseFunctionCallArgumentsDeltaStreamingEvent
    | ResponseFunctionCallArgumentsDoneStreamingEvent
    | ResponseReasoningSummaryPartAddedStreamingEvent
    | ResponseReasoningSummaryPartDoneStreamingEvent
    | ResponseReasoningSummaryDeltaStreamingEvent
    | ResponseReasoningSummaryDoneStreamingEvent
    | ResponseReasoningDeltaStreamingEvent
    | ResponseReasoningDoneStreamingEvent
    | ResponseOutputTextAnnotationAddedStreamingEvent
)

ResponseInputParam = (
    Input
    | list[
        Annotated[
            ItemReferenceParam
            | ReasoningItemParam
            | UserMessageItemParam
            | SystemMessageItemParam
            | DeveloperMessageItemParam
            | AssistantMessageItemParam
            | FunctionCallItemParam
            | FunctionCallOutputItemParam
            | dict[str, Any],
            Field(),
        ]
    ]
)

# Backward-compatible aliases
OpenResponsesReasoningConfig = ReasoningParam
OpenResponsesReasoningItem = ReasoningItemParam
FunctionTool = FunctionToolParam
ReasoningDeltaEvent = ResponseReasoningDeltaStreamingEvent
ReasoningSummaryDeltaEvent = ResponseReasoningSummaryDeltaStreamingEvent
OpenResponsesResponse = ResponseResource
OpenResponsesRequestBody = CreateResponseBody
GeneratedReasoningEffort = ReasoningEffortEnum


# =============================================================================
# Custom Types - not yet in OpenResponses spec
# =============================================================================


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for models that support reasoning."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class MCPToolApproval(str, Enum):
    """Approval mode for MCP tool execution."""

    NEVER = "never"
    ALWAYS = "always"


class MCPTool(BaseModel):
    """Model Context Protocol (MCP) tool definition.

    MCP tools are remote server-hosted tools that the model can invoke.
    Note: MCP tool support is an extension not yet in the OpenResponses spec.

    See: https://modelcontextprotocol.io/
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["mcp"] = "mcp"
    server_label: str
    server_url: str
    allowed_tools: list[str] | None = None
    require_approval: MCPToolApproval | str = MCPToolApproval.NEVER
    headers: dict[str, str] | None = None


class ItemStatus(str, Enum):
    """Lifecycle status for OpenResponses items."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class OpenResponsesStreamEventType(str, Enum):
    """Semantic streaming event types in OpenResponses."""

    RESPONSE_CREATED = "response.created"
    RESPONSE_IN_PROGRESS = "response.in_progress"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_FAILED = "response.failed"
    OUTPUT_ITEM_ADDED = "response.output_item.added"
    OUTPUT_ITEM_DONE = "response.output_item.done"
    OUTPUT_TEXT_DELTA = "response.output_text.delta"
    OUTPUT_TEXT_DONE = "response.output_text.done"
    REASONING_DELTA = "response.reasoning.delta"
    REASONING_DONE = "response.reasoning.done"
    REASONING_SUMMARY_DELTA = "response.reasoning_summary_text.delta"
    REASONING_SUMMARY_DONE = "response.reasoning_summary_text.done"
    FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    CONTENT_PART_ADDED = "response.content_part.added"
    CONTENT_PART_DONE = "response.content_part.done"


class ResponsesParams(BaseModel):
    """Normalized parameters for OpenResponses API.

    Used internally to pass structured parameters from the public API
    to provider implementations.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    input: str | ResponseInputParam
    instructions: str | None = None
    max_tool_calls: int | None = None
    text: Any | None = None
    tools: Sequence[dict[str, Any] | FunctionToolParam | MCPTool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    response_format: dict[str, Any] | type[BaseModel] | None = None
    stream: bool | None = None
    parallel_tool_calls: bool | None = None
    top_logprobs: int | None = None
    stream_options: dict[str, Any] | None = None
    reasoning: ReasoningParam | dict[str, Any] | None = None
    previous_response_id: str | None = None
    truncation: Literal["auto", "disabled"] | None = None
    service_tier: Literal["auto", "default", "flex", "priority"] | None = None
    include: list[str] | None = None
    metadata: dict[str, str] | None = Field(default=None, max_length=16)


# =============================================================================
# Helper Functions
# =============================================================================


def get_output_text(response: Response) -> str:
    """Extract the output text from a Response.

    Args:
        response: The Response object.

    Returns:
        The concatenated text from all output_text content blocks.
    """
    texts: list[str] = []
    for output in response.output:
        if getattr(output, "type", None) == "message":
            for content in getattr(output, "content", []):
                if getattr(content, "type", None) == "output_text":
                    texts.append(getattr(content, "text", ""))
    return "".join(texts)
