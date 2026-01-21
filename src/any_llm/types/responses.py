"""OpenResponses API types.

This module provides types for the OpenResponses API specification, which extends
OpenAI's Responses API with additional features for agentic AI systems.

OpenResponses is an open-source specification for building multi-provider,
interoperable LLM interfaces. It adds:
- Reasoning content (raw, encrypted, or summary)
- Semantic streaming events
- MCP (Model Context Protocol) tool support
- Provider routing via model:provider syntax

See: https://www.openresponses.org/specification

All types are based on the auto-generated openresponses_generated.py from the
OpenResponses OpenAPI specification.
"""

from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from any_llm.types.openresponses_generated import (
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
# Core Response Types (from OpenResponses spec)
# =============================================================================

# The main Response type - uses OpenResponses ResponseResource
Response = ResponseResource

# Union of all streaming event types from OpenResponses spec
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

# Input parameter type - matches CreateResponseBody.input
# Also accepts dict for flexibility with untyped input
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

# Re-export generated types with backward-compatible aliases
OpenResponsesReasoningConfig = ReasoningParam
OpenResponsesReasoningItem = ReasoningItemParam
FunctionTool = FunctionToolParam
ReasoningDeltaEvent = ResponseReasoningDeltaStreamingEvent
ReasoningSummaryDeltaEvent = ResponseReasoningSummaryDeltaStreamingEvent
OpenResponsesResponse = ResponseResource
OpenResponsesRequestBody = CreateResponseBody

# Export the generated enum for users who prefer StrEnum over Enum
GeneratedReasoningEffort = ReasoningEffortEnum


# =============================================================================
# OpenResponses Reasoning Types
# =============================================================================


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for models that support reasoning.

    Controls the depth and quality of reasoning traces produced by the model.
    Higher effort levels may increase latency but produce more thorough reasoning.

    Note: Consider using ReasoningEffortEnum from openresponses_generated.py for
    new code.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


# =============================================================================
# OpenResponses MCP Tool Types (not in OpenResponses spec yet)
# =============================================================================


class MCPToolApproval(str, Enum):
    """Approval mode for MCP tool execution."""

    NEVER = "never"
    ALWAYS = "always"


class MCPTool(BaseModel):
    """Model Context Protocol (MCP) tool definition for OpenResponses.

    MCP tools are remote server-hosted tools that the model can invoke.
    The server handles tool execution and returns results to the model.

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


# =============================================================================
# OpenResponses Item Types
# =============================================================================


class ItemStatus(str, Enum):
    """Lifecycle status for OpenResponses items."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


# =============================================================================
# OpenResponses Streaming Event Types
# =============================================================================


class OpenResponsesStreamEventType(str, Enum):
    """Semantic streaming event types in OpenResponses.

    OpenResponses uses semantic events rather than raw text deltas,
    allowing clients to understand the meaning of each update.
    """

    # Response lifecycle events
    RESPONSE_CREATED = "response.created"
    RESPONSE_IN_PROGRESS = "response.in_progress"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_FAILED = "response.failed"

    # Output item events
    OUTPUT_ITEM_ADDED = "response.output_item.added"
    OUTPUT_ITEM_DONE = "response.output_item.done"

    # Text content events
    OUTPUT_TEXT_DELTA = "response.output_text.delta"
    OUTPUT_TEXT_DONE = "response.output_text.done"

    # Reasoning events (OpenResponses extension)
    REASONING_DELTA = "response.reasoning.delta"
    REASONING_DONE = "response.reasoning.done"
    REASONING_SUMMARY_DELTA = "response.reasoning_summary_text.delta"
    REASONING_SUMMARY_DONE = "response.reasoning_summary_text.done"

    # Tool events
    FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"

    # Content part events
    CONTENT_PART_ADDED = "response.content_part.added"
    CONTENT_PART_DONE = "response.content_part.done"


class OpenResponsesStreamEvent(BaseModel):
    """Base class for OpenResponses streaming events."""

    model_config = ConfigDict(extra="allow")

    type: str
    sequence_number: int | None = None


# =============================================================================
# OpenResponses Request Parameters
# =============================================================================


class ResponsesParams(BaseModel):
    """Normalized parameters for OpenResponses API.

    This model is used internally to pass structured parameters from the public
    API layer to provider implementations.

    Note: For the full OpenResponses request schema, see CreateResponseBody
    in openresponses_generated.py.

    See: https://www.openresponses.org/specification
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

    # OpenResponses extensions
    previous_response_id: str | None = None
    truncation: Literal["auto", "disabled"] | None = None
    service_tier: Literal["auto", "default", "flex", "priority"] | None = None
    include: list[str] | None = None
    metadata: dict[str, str] | None = Field(default=None, max_length=16)


# =============================================================================
# Output text helper
# =============================================================================


def get_output_text(response: Response) -> str:
    """Extract the output text from a Response.

    This is a helper function that mimics the output_text property from
    the OpenAI SDK's Response type.

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
