"""Pydantic models for acceptance test server requests and responses."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ScenarioID(str, Enum):
    """Available test scenarios."""

    BASIC_COMPLETION = "basic_completion"
    TOOL_CALLS = "tool_calls"
    TOOL_RESPONSE = "tool_response"
    STREAMING = "streaming"
    STRUCTURED_OUTPUT = "structured_output"
    MULTI_TURN = "multi_turn"
    SYSTEM_MESSAGE = "system_message"
    IMAGE_CONTENT = "image_content"
    TEMPERATURE_PARAMS = "temperature_params"


MODEL_TO_SCENARIO: dict[str, ScenarioID] = {
    "test-basic": ScenarioID.BASIC_COMPLETION,
    "test-tools": ScenarioID.TOOL_CALLS,
    "test-tool-response": ScenarioID.TOOL_RESPONSE,
    "test-stream": ScenarioID.STREAMING,
    "test-structured": ScenarioID.STRUCTURED_OUTPUT,
    "test-multi-turn": ScenarioID.MULTI_TURN,
    "test-system": ScenarioID.SYSTEM_MESSAGE,
    "test-image": ScenarioID.IMAGE_CONTENT,
    "test-params": ScenarioID.TEMPERATURE_PARAMS,
}


class TextContentPart(BaseModel):
    """Text content part in a message."""

    type: Literal["text"]
    text: str


class ImageUrlDetail(BaseModel):
    """Image URL details."""

    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ImageContentPart(BaseModel):
    """Image content part in a message."""

    type: Literal["image_url"]
    image_url: ImageUrlDetail


ContentPart = TextContentPart | ImageContentPart


class FunctionDefinition(BaseModel):
    """Function definition within a tool."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class FunctionTool(BaseModel):
    """Function tool definition."""

    type: Literal["function"]
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """Function call in a tool call."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call from assistant message."""

    id: str
    type: Literal["function"]
    function: FunctionCall


class Message(BaseModel):
    """Chat message in the request."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class JsonSchemaFormat(BaseModel):
    """JSON schema response format."""

    name: str
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    description: str | None = None
    strict: bool | None = None


class ResponseFormat(BaseModel):
    """Response format specification."""

    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaFormat | None = None


class StreamOptions(BaseModel):
    """Stream options for streaming requests."""

    include_usage: bool | None = None


class ToolChoiceFunction(BaseModel):
    """Function specification in tool choice."""

    name: str


class ToolChoiceObject(BaseModel):
    """Named tool choice object."""

    type: Literal["function"]
    function: ToolChoiceFunction


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[Message]
    tools: list[FunctionTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | ToolChoiceObject | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    response_format: ResponseFormat | None = None
    stream: bool | None = None
    stream_options: StreamOptions | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    user: str | None = None
    parallel_tool_calls: bool | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None


class ValidationError(BaseModel):
    """A single validation error."""

    field: str
    message: str
    expected: Any = None
    actual: Any = None


class ValidationResult(BaseModel):
    """Result of validating a request against a scenario."""

    scenario: ScenarioID
    passed: bool
    errors: list[ValidationError] = Field(default_factory=list)
    request_id: str | None = None
    timestamp: float | None = None
    test_run_id: str | None = None


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    """Message in a completion choice."""

    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChoiceMessage
    finish_reason: Literal["stop", "tool_calls", "length"]


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo


class ChunkDelta(BaseModel):
    """Delta in a streaming chunk."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChunkChoice(BaseModel):
    """Choice in a streaming chunk."""

    index: int
    delta: ChunkDelta
    finish_reason: Literal["stop", "tool_calls", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: UsageInfo | None = None


class ScenarioInfo(BaseModel):
    """Information about a test scenario."""

    id: ScenarioID
    name: str
    description: str
    model_name: str
    required_fields: list[str]
    optional_fields: list[str]


class TestRun(BaseModel):
    """A test run groups multiple validation results."""

    id: str
    created_at: float
    description: str | None = None
    metadata: dict[str, Any] | None = None


class RequestInfo(BaseModel):
    """Information about a stored request."""

    id: int
    test_run_id: str
    request_id: str
    scenario: str
    timestamp: float
    request_body: dict[str, Any] | None = None


class TestRunSummary(BaseModel):
    """Summary of a test run."""

    test_run_id: str | None = None
    total: int
    by_scenario: dict[str, int]


class ResultsResponse(BaseModel):
    """Response containing request tracking results."""

    test_run_id: str | None = None
    total: int
    requests: list[RequestInfo]
