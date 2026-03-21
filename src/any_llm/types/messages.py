from typing import Any

from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock as AnthropicTextBlock
from anthropic.types import ThinkingBlock as AnthropicThinkingBlock
from anthropic.types import ToolUseBlock as AnthropicToolUseBlock
from anthropic.types import Usage as AnthropicUsage
from pydantic import BaseModel, ConfigDict

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


class MessageStreamEvent(BaseModel):
    """Server-sent event for Messages API streaming."""

    type: str
    """Event type (message_start, content_block_start, content_block_delta, content_block_stop, message_delta, message_stop)"""

    index: int | None = None
    """Content block index"""

    content_block: ContentBlock | None = None
    """Content block (for content_block_start)"""

    delta: dict[str, Any] | None = None
    """Delta update"""

    message: MessageResponse | None = None
    """Full message (for message_start)"""

    usage: MessageUsage | None = None
    """Usage information (for message_delta)"""


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
