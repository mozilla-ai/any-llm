from typing import TYPE_CHECKING, Any

from anthropic.types import ContentBlock as AnthropicContentBlock
from anthropic.types import InputJSONDelta, RawContentBlockDelta, TextDelta, ThinkingDelta
from anthropic.types import Message as AnthropicMessage
from anthropic.types import MessageDeltaUsage as AnthropicMessageDeltaUsage
from anthropic.types import RawContentBlockDeltaEvent as AnthropicContentBlockDeltaEvent
from anthropic.types import RawContentBlockStartEvent as AnthropicContentBlockStartEvent
from anthropic.types import RawContentBlockStopEvent as AnthropicContentBlockStopEvent
from anthropic.types import RawMessageDeltaEvent as AnthropicMessageDeltaEvent
from anthropic.types import RawMessageStartEvent as AnthropicMessageStartEvent
from anthropic.types import RawMessageStopEvent as AnthropicMessageStopEvent
from anthropic.types import TextBlock as AnthropicTextBlock
from anthropic.types import ThinkingBlock as AnthropicThinkingBlock
from anthropic.types import ToolUseBlock as AnthropicToolUseBlock
from anthropic.types import Usage as AnthropicUsage
from anthropic.types.beta import (
    BetaCompactionBlock,
    BetaCompactionContentBlockDelta,
    BetaContentBlock,
    BetaIterationsUsage,
    BetaStopReason,
)
from anthropic.types.beta.beta_context_management_response import BetaContextManagementResponse
from anthropic.types.beta.parsed_beta_message import ParsedBetaMessage, ParsedBetaTextBlock
from anthropic.types.parsed_message import ParsedMessage, ParsedTextBlock
from anthropic.types.raw_message_delta_event import Delta as AnthropicMessageDelta
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from anthropic.types.beta.beta_diagnostics import BetaDiagnostics
else:
    try:
        from anthropic.types.beta.beta_diagnostics import BetaDiagnostics
    except ModuleNotFoundError:
        # BetaDiagnostics was added in anthropic 0.102.0, while any-llm supports 0.83.0.
        class _BetaDiagnosticsFallback(BaseModel):
            model_config = ConfigDict(extra="allow", from_attributes=True)

            cache_miss_reason: dict[str, Any] | None = None

        BetaDiagnostics = _BetaDiagnosticsFallback

__all__ = [
    "BetaContextManagementResponse",
    "BetaDiagnostics",
    "CompactionBlock",
    "CompactionDelta",
    "ContentBlock",
    "ContentBlockDeltaEvent",
    "ContentBlockStartEvent",
    "ContentBlockStopEvent",
    "InputJSONDelta",
    "MessageContentBlock",
    "MessageDelta",
    "MessageDeltaEvent",
    "MessageDeltaUsage",
    "MessageResponse",
    "MessageStartEvent",
    "MessageStopEvent",
    "MessageStreamEvent",
    "MessageUsage",
    "MessagesParams",
    "ParsedBetaMessage",
    "ParsedBetaTextBlock",
    "ParsedMessage",
    "ParsedTextBlock",
    "StopReason",
    "TextBlock",
    "TextDelta",
    "ThinkingBlock",
    "ThinkingDelta",
    "ToolUseBlock",
]

StopReason = BetaStopReason


class MessageUsage(AnthropicUsage):
    iterations: BetaIterationsUsage | None = None


TextBlock = AnthropicTextBlock

ToolUseBlock = AnthropicToolUseBlock


class ThinkingBlock(AnthropicThinkingBlock):
    signature: str = ""


CompactionBlock = BetaCompactionBlock
CompactionDelta = BetaCompactionContentBlockDelta

ContentBlock = TextBlock | ToolUseBlock | ThinkingBlock | CompactionBlock

MessageContentBlock = AnthropicContentBlock | BetaContentBlock


class MessageResponse(AnthropicMessage):
    content: list[MessageContentBlock]  # type: ignore[assignment]
    stop_reason: StopReason | None = None  # type: ignore[assignment]
    usage: MessageUsage
    context_management: BetaContextManagementResponse | None = None
    diagnostics: BetaDiagnostics | None = None


class MessageDelta(AnthropicMessageDelta):
    stop_reason: StopReason | None = None  # type: ignore[assignment]


class MessageDeltaUsage(AnthropicMessageDeltaUsage):
    iterations: BetaIterationsUsage | None = None


class MessageStartEvent(AnthropicMessageStartEvent):
    message: MessageResponse


class MessageDeltaEvent(AnthropicMessageDeltaEvent):
    delta: MessageDelta
    usage: MessageDeltaUsage
    context_management: BetaContextManagementResponse | None = None


class MessageStopEvent(AnthropicMessageStopEvent):
    pass


class ContentBlockStartEvent(AnthropicContentBlockStartEvent):
    content_block: MessageContentBlock  # type: ignore[assignment]


class ContentBlockDeltaEvent(AnthropicContentBlockDeltaEvent):
    delta: RawContentBlockDelta | CompactionDelta  # type: ignore[assignment]


class ContentBlockStopEvent(AnthropicContentBlockStopEvent):
    content_block: MessageContentBlock | None = None


MessageStreamEvent = (
    MessageStartEvent
    | MessageDeltaEvent
    | MessageStopEvent
    | ContentBlockStartEvent
    | ContentBlockDeltaEvent
    | ContentBlockStopEvent
)


class MessagesParams(BaseModel):
    """Normalized parameters for Anthropic Messages API."""

    model_config = ConfigDict(extra="forbid")

    model: str
    """Model identifier"""

    messages: list[dict[str, Any]]
    """List of messages for the conversation"""

    max_tokens: int
    """Maximum number of tokens to generate (required by Anthropic API)"""

    system: str | list[dict[str, Any]] | None = None
    """System prompt (string or list of content blocks with optional cache_control)"""

    temperature: float | None = None
    """Controls randomness in the response (0.0 to 1.0)"""

    top_p: float | None = None
    """Controls diversity via nucleus sampling"""

    top_k: int | None = None
    """Only sample from the top K options for each subsequent token"""

    stream: bool | None = None
    """Whether to stream the response"""

    stop_sequences: list[str] | None = None
    """Custom text sequences that will cause the model to stop generating"""

    tools: list[dict[str, Any]] | None = None
    """List of tools in Anthropic format ({name, description, input_schema})"""

    tool_choice: dict[str, Any] | None = None
    """Controls which tool the model uses"""

    metadata: dict[str, Any] | None = None
    """Request metadata"""

    thinking: dict[str, Any] | None = None
    """Thinking/reasoning configuration"""

    cache_control: dict[str, Any] | None = None
    """Cache control configuration for prompt caching"""

    context_management: dict[str, Any] | None = None
    """Anthropic context management configuration"""

    betas: list[str] | None = None
    """Anthropic beta identifiers"""

    output_format: type | dict[str, Any] | None = None
    """Structured output, mirroring Anthropic's ``messages.parse``/``output_config``.

    Either a Pydantic ``BaseModel`` subclass or dataclass **type**, or a raw Anthropic
    ``output_config`` **dict** (e.g. ``{"format": {"type": "json_schema", "schema": {...}}}``)
    for non-Pydantic JSON schemas. A type goes to native ``messages.parse`` on Anthropic; a
    dict is passed through to native ``messages.create(output_config=...)``. Other providers
    route either form through the completion bridge. The result is Anthropic's ``ParsedMessage``:
    its ``parsed_output`` holds the typed object for a type, or the parsed JSON (plain
    ``dict``/``list``) for a raw schema.
    """
