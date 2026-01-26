from typing import Any

from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseInputParam as OpenAIResponseInputParam
from openai.types.responses import ResponseOutputMessage as OpenAIResponseOutputMessage
from openai.types.responses import ResponseStreamEvent as OpenAIResponseStreamEvent
from pydantic import BaseModel, ConfigDict

Response = OpenAIResponse
ResponseStreamEvent = OpenAIResponseStreamEvent
ResponseOutputMessage = OpenAIResponseOutputMessage
ResponseInputParam = OpenAIResponseInputParam


class ResponsesParams(BaseModel):
    """Normalized parameters for responses API.

    This model is used internally to pass structured parameters from the public
    API layer to provider implementations, avoiding very long function
    signatures while keeping type safety.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    """Model identifier (e.g., 'mistral-small-latest')"""

    input: str | ResponseInputParam
    """The input payload accepted by provider's Responses API.
        For OpenAI-compatible providers, this is typically a list mixing
        text, images, and tool instructions, or a dict per OpenAI spec.
    """

    instructions: str | None = None

    max_tool_calls: int | None = None

    text: Any | None = None

    tools: list[dict[str, Any]] | None = None
    """List of tools for tool calling. Should be converted to OpenAI tool format dicts"""

    tool_choice: str | dict[str, Any] | None = None
    """Controls which tools the model can call"""

    temperature: float | None = None
    """Controls randomness in the response (0.0 to 2.0)"""

    top_p: float | None = None
    """Controls diversity via nucleus sampling (0.0 to 1.0)"""

    max_output_tokens: int | None = None
    """Maximum number of tokens to generate"""

    response_format: dict[str, Any] | type[BaseModel] | None = None
    """Format specification for the response"""

    stream: bool | None = None
    """Whether to stream the response"""

    parallel_tool_calls: bool | None = None
    """Whether to allow parallel tool calls"""

    top_logprobs: int | None = None
    """Number of top alternatives to return when logprobs are requested"""

    stream_options: dict[str, Any] | None = None
    """Additional options controlling streaming behavior"""

    reasoning: dict[str, Any] | None = None
    """Configuration options for reasoning models."""
