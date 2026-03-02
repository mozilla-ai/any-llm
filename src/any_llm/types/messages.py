from typing import Any

from pydantic import BaseModel, ConfigDict


class MessagesParams(BaseModel):
    """Normalized parameters for Anthropic Messages API.

    This model is used internally to pass structured parameters from the public
    API layer to provider implementations. Matches the Anthropic Messages API format.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    """Model identifier"""

    messages: list[dict[str, Any]]
    """List of messages for the conversation"""

    max_tokens: int
    """Maximum number of tokens to generate (required by Anthropic API)"""

    system: str | None = None
    """System prompt"""

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


class MessageContentBlock(BaseModel):
    """Content block in a Messages API response."""

    type: str
    """Block type: 'text', 'tool_use', or 'thinking'"""

    text: str | None = None
    """Text content (for type='text')"""

    id: str | None = None
    """Tool use ID (for type='tool_use')"""

    name: str | None = None
    """Tool name (for type='tool_use')"""

    input: dict[str, Any] | None = None
    """Tool input (for type='tool_use')"""

    thinking: str | None = None
    """Thinking content (for type='thinking')"""


class MessageUsage(BaseModel):
    """Token usage information for Messages API."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class MessageResponse(BaseModel):
    """Full response from the Messages API."""

    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[MessageContentBlock]
    model: str
    stop_reason: str | None = None
    usage: MessageUsage


class MessageStreamEvent(BaseModel):
    """Server-sent event for Messages API streaming."""

    type: str
    """Event type (message_start, content_block_start, content_block_delta, content_block_stop, message_delta, message_stop)"""

    index: int | None = None
    """Content block index"""

    content_block: MessageContentBlock | None = None
    """Content block (for content_block_start)"""

    delta: dict[str, Any] | None = None
    """Delta update"""

    message: MessageResponse | None = None
    """Full message (for message_start)"""

    usage: MessageUsage | None = None
    """Usage information (for message_delta)"""
