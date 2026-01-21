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
"""

from collections.abc import Sequence
from enum import Enum
from typing import Any, Literal

from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseInputParam as OpenAIResponseInputParam
from openai.types.responses import ResponseOutputMessage as OpenAIResponseOutputMessage
from openai.types.responses import ResponseStreamEvent as OpenAIResponseStreamEvent
from pydantic import BaseModel, ConfigDict, Field

# Re-export OpenAI types for backward compatibility
Response = OpenAIResponse
ResponseStreamEvent = OpenAIResponseStreamEvent
ResponseOutputMessage = OpenAIResponseOutputMessage
ResponseInputParam = OpenAIResponseInputParam


# =============================================================================
# OpenResponses Reasoning Types
# =============================================================================


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for models that support reasoning.

    Controls the depth and quality of reasoning traces produced by the model.
    Higher effort levels may increase latency but produce more thorough reasoning.
    """

    NONE = "none"
    """No reasoning - model responds directly without reasoning traces."""

    LOW = "low"
    """Minimal reasoning - quick, shallow reasoning traces."""

    MEDIUM = "medium"
    """Balanced reasoning - moderate depth and latency."""

    HIGH = "high"
    """Deep reasoning - thorough reasoning traces, higher latency."""

    XHIGH = "xhigh"
    """Maximum reasoning - most thorough reasoning, highest latency."""


class OpenResponsesReasoningConfig(BaseModel):
    """Configuration for reasoning behavior in OpenResponses.

    Controls how the model produces and returns reasoning traces.
    """

    model_config = ConfigDict(extra="allow")

    effort: ReasoningEffort | str | None = None
    """The effort level for reasoning. Higher levels produce more thorough traces."""

    max_tokens: int | None = None
    """Maximum tokens to allocate for reasoning content."""

    encrypted: bool | None = None
    """Whether to return encrypted reasoning content (provider-specific)."""


class OpenResponsesReasoningItem(BaseModel):
    """A reasoning item in an OpenResponses response.

    Reasoning items contain the model's chain-of-thought or reasoning traces.
    OpenResponses formalizes three optional fields for reasoning content.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    """Unique identifier for this reasoning item."""

    type: Literal["reasoning"] = "reasoning"
    """Item type identifier."""

    status: Literal["in_progress", "completed", "failed", "incomplete"] | None = None
    """Lifecycle state of this reasoning item."""

    content: str | None = None
    """Raw reasoning traces from the model.

    Contains the unprocessed chain-of-thought content. May be omitted
    if the provider only returns encrypted or summarized reasoning.
    """

    encrypted_content: str | None = None
    """Provider-specific encrypted reasoning content.

    Some providers protect reasoning traces with encryption. This field
    contains opaque, provider-specific content that cannot be decoded
    by the client but can be passed back to the provider.
    """

    summary: str | None = None
    """Sanitized, human-readable summary of the reasoning.

    A natural-language explanation of the reasoning process, suitable
    for display to end users. May be sanitized to remove sensitive content.
    """


# =============================================================================
# OpenResponses MCP Tool Types
# =============================================================================


class MCPToolApproval(str, Enum):
    """Approval mode for MCP tool execution."""

    NEVER = "never"
    """Never require approval - tools execute automatically."""

    ALWAYS = "always"
    """Always require approval before tool execution."""


class MCPTool(BaseModel):
    """Model Context Protocol (MCP) tool definition for OpenResponses.

    MCP tools are remote server-hosted tools that the model can invoke.
    The server handles tool execution and returns results to the model.

    See: https://modelcontextprotocol.io/
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["mcp"] = "mcp"
    """Tool type identifier - must be 'mcp' for MCP tools."""

    server_label: str
    """Human-readable label for the MCP server."""

    server_url: str
    """URL of the MCP server endpoint."""

    allowed_tools: list[str] | None = None
    """List of tool names the model is allowed to invoke on this server.

    If None or empty, all tools on the server may be used.
    """

    require_approval: MCPToolApproval | str = MCPToolApproval.NEVER
    """Whether tool calls require user approval before execution."""

    headers: dict[str, str] | None = None
    """Additional headers to send with MCP requests."""


class FunctionTool(BaseModel):
    """Function tool definition for OpenResponses.

    Standard function tools that the client executes locally.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    """Tool type identifier."""

    name: str
    """Function name that the model will use to invoke this tool."""

    description: str | None = None
    """Description of what the function does, used by the model."""

    parameters: dict[str, Any] | None = None
    """JSON Schema defining the function's parameters."""

    strict: bool | None = None
    """Whether to enforce strict parameter validation."""


# =============================================================================
# OpenResponses Item Types
# =============================================================================


class ItemStatus(str, Enum):
    """Lifecycle status for OpenResponses items.

    Items are the fundamental unit of context in OpenResponses,
    representing atomic units of model output, tool invocation, or reasoning state.
    """

    IN_PROGRESS = "in_progress"
    """Model is currently emitting tokens for this item."""

    COMPLETED = "completed"
    """Model has finished emission; no further updates will occur."""

    FAILED = "failed"
    """Item processing failed due to an error."""

    INCOMPLETE = "incomplete"
    """Model exhausted token budget before completing (terminal state)."""


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
    """Base class for OpenResponses streaming events.

    Each streaming event contains a type and sequence number for ordering.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    """Event type identifier (e.g., 'response.output_text.delta')."""

    sequence_number: int | None = None
    """Sequence number for ordering events."""


class ReasoningDeltaEvent(OpenResponsesStreamEvent):
    """Streaming event for reasoning content deltas."""

    type: Literal["response.reasoning.delta"] = "response.reasoning.delta"
    delta: str
    """Incremental reasoning content."""

    item_id: str | None = None
    """ID of the reasoning item being updated."""


class ReasoningSummaryDeltaEvent(OpenResponsesStreamEvent):
    """Streaming event for reasoning summary deltas."""

    type: Literal["response.reasoning_summary_text.delta"] = "response.reasoning_summary_text.delta"
    delta: str
    """Incremental reasoning summary content."""

    item_id: str | None = None
    """ID of the reasoning item being summarized."""


# =============================================================================
# OpenResponses Request Parameters
# =============================================================================


class ResponsesParams(BaseModel):
    """Normalized parameters for OpenResponses API.

    This model is used internally to pass structured parameters from the public
    API layer to provider implementations, avoiding very long function
    signatures while keeping type safety.

    Supports both standard OpenAI Responses API parameters and OpenResponses
    extensions like MCP tools and reasoning configuration.

    See: https://www.openresponses.org/specification
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    """Model identifier.

    For OpenResponses router, use 'model_id:provider' format for routing
    (e.g., 'moonshotai/Kimi-K2-Instruct:groq').
    """

    input: str | ResponseInputParam
    """The input payload for the request.

    Can be a simple string (interpreted as user message) or a list of
    content items mixing text, images, and tool instructions.
    """

    instructions: str | None = None
    """System or developer instructions for the model.

    Inserted into the model's context to guide behavior.
    """

    max_tool_calls: int | None = None
    """Maximum number of tool calls allowed in this response.

    Applies across all tool types (functions, MCP, etc.).
    """

    text: Any | None = None
    """Configuration for text response format."""

    tools: Sequence[dict[str, Any] | FunctionTool | MCPTool] | None = None
    """List of tools available to the model.

    Supports both function tools (client-executed) and MCP tools
    (server-executed via Model Context Protocol).

    Example:
        tools=[
            FunctionTool(name="search", description="Search the web"),
            MCPTool(server_label="github", server_url="https://mcp.github.com"),
        ]
    """

    tool_choice: str | dict[str, Any] | None = None
    """Controls which tools the model can call.

    - "auto": Model decides whether to call tools
    - "required": Model must call at least one tool
    - "none": Model cannot call tools
    - {"type": "function", "name": "..."}: Force specific tool
    """

    temperature: float | None = None
    """Controls randomness in the response (0.0 to 2.0)."""

    top_p: float | None = None
    """Controls diversity via nucleus sampling (0.0 to 1.0)."""

    max_output_tokens: int | None = None
    """Maximum number of tokens to generate."""

    response_format: dict[str, Any] | type[BaseModel] | None = None
    """Format specification for structured output.

    Use JSON schema to enforce response structure.
    """

    stream: bool | None = None
    """Whether to stream response events.

    When True, returns semantic events like 'response.output_text.delta'.
    """

    parallel_tool_calls: bool | None = None
    """Whether to allow parallel tool calls."""

    top_logprobs: int | None = None
    """Number of top token alternatives to return when logprobs are requested."""

    stream_options: dict[str, Any] | None = None
    """Additional options controlling streaming behavior."""

    reasoning: OpenResponsesReasoningConfig | dict[str, Any] | None = None
    """Configuration for reasoning models.

    Controls reasoning effort level and output format.

    Example:
        reasoning=OpenResponsesReasoningConfig(effort=ReasoningEffort.HIGH)
        # or
        reasoning={"effort": "high"}
    """

    # OpenResponses extensions
    previous_response_id: str | None = None
    """ID of a previous response to continue from.

    Enables multi-turn conversations without resending full context.
    Server loads prior input/output and concatenates with new input.
    """

    truncation: Literal["auto", "disabled"] | None = None
    """Context truncation behavior.

    - "auto": Server may shorten context to fit model window
    - "disabled": Server must fail if context exceeds limit
    """

    service_tier: Literal["auto", "default", "flex", "priority"] | None = None
    """Hint to provider about request priority."""

    include: list[str] | None = None
    """Additional content to include in response (e.g., 'reasoning.encrypted_content')."""

    metadata: dict[str, str] | None = Field(default=None, max_length=16)
    """Custom metadata for the request (max 16 key-value pairs)."""
