from typing import Any

from anthropic.types import InputJSONDelta, MessageDeltaUsage, TextDelta, ThinkingDelta
from anthropic.types import Message as AnthropicMessage
from anthropic.types import RawContentBlockDeltaEvent as ContentBlockDeltaEvent
from anthropic.types import RawContentBlockStartEvent as ContentBlockStartEvent
from anthropic.types import RawContentBlockStopEvent as ContentBlockStopEvent
from anthropic.types import RawMessageDeltaEvent as MessageDeltaEvent
from anthropic.types import RawMessageStartEvent as MessageStartEvent
from anthropic.types import RawMessageStopEvent as MessageStopEvent
from anthropic.types import TextBlock as AnthropicTextBlock
from anthropic.types import ThinkingBlock as AnthropicThinkingBlock
from anthropic.types import ToolUseBlock as AnthropicToolUseBlock
from anthropic.types import Usage as AnthropicUsage
from anthropic.types.raw_message_delta_event import Delta as MessageDelta
from pydantic import BaseModel, ConfigDict

__all__ = [
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
    "TextBlock",
    "TextDelta",
    "ThinkingBlock",
    "ThinkingDelta",
    "ToolUseBlock",
]

MessageUsage = AnthropicUsage

TextBlock = AnthropicTextBlock

ToolUseBlock = AnthropicToolUseBlock


class ThinkingBlock(AnthropicThinkingBlock):
    signature: str = ""


ContentBlock = TextBlock | ToolUseBlock | ThinkingBlock

MessageContentBlock = ContentBlock


class MessageResponse(AnthropicMessage):
    content: list[ContentBlock]  # type: ignore[assignment]
    stop_reason: str | None = None  # type: ignore[assignment]


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
