from typing import Any, Literal

from openai.types import CreateEmbeddingResponse as OpenAICreateEmbeddingResponse
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion import Choice as OpenAIChoice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as OpenAIChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta as OpenAIChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall as OpenAIChoiceDeltaToolCall
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCallFunction as OpenAIChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage as OpenAIChatCompletionMessage
from openai.types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall as OpenAIChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as OpenAIChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import Function as OpenAIFunction
from openai.types.completion_usage import CompletionUsage as OpenAICompletionUsage
from openai.types.create_embedding_response import Usage as OpenAIUsage
from openai.types.embedding import Embedding as OpenAIEmbedding
from pydantic import BaseModel, ConfigDict, field_validator

# See https://github.com/mozilla-ai/any-llm/issues/95:
# OpenAI Completion API doesn't include reasoning information, so we need to extend the openai type


class Reasoning(BaseModel):
    content: str


class ChatCompletionMessage(OpenAIChatCompletionMessage):
    reasoning: Reasoning | None = None


class Choice(OpenAIChoice):
    message: ChatCompletionMessage


class ChatCompletion(OpenAIChatCompletion):
    choices: list[Choice]  # type: ignore[assignment]


class ChoiceDelta(OpenAIChoiceDelta):
    reasoning: Reasoning | None = None


class ChunkChoice(OpenAIChunkChoice):
    delta: ChoiceDelta


class ChatCompletionChunk(OpenAIChatCompletionChunk):
    choices: list[ChunkChoice]  # type: ignore[assignment]


ChatCompletionMessageFunctionToolCall = OpenAIChatCompletionMessageFunctionToolCall
ChatCompletionMessageToolCall = OpenAIChatCompletionMessageFunctionToolCall | OpenAIChatCompletionMessageToolCall
Function = OpenAIFunction
CompletionUsage = OpenAICompletionUsage
CreateEmbeddingResponse = OpenAICreateEmbeddingResponse
Embedding = OpenAIEmbedding
Usage = OpenAIUsage
ChoiceDeltaToolCall = OpenAIChoiceDeltaToolCall
ChoiceDeltaToolCallFunction = OpenAIChoiceDeltaToolCallFunction


class CompletionParams(BaseModel):
    """Normalized parameters for chat completions.

    This model is used internally to pass structured parameters from the public
    API layer to provider implementations, avoiding very long function
    signatures while keeping type safety.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    """Model identifier (e.g., 'mistral-small-latest')"""

    messages: list[dict[str, Any]]
    """List of messages for the conversation"""

    @field_validator("messages")
    def check_messages_not_empty(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:  # noqa: N805
        if not v:
            msg = "The `messages` list cannot be empty."
            raise ValueError(msg)
        return v

    tools: list[dict[str, Any]] | None = None
    """List of tools for tool calling. Should be converted to OpenAI tool format dicts"""

    tool_choice: str | dict[str, Any] | None = None
    """Controls which tools the model can call"""

    temperature: float | None = None
    """Controls randomness in the response (0.0 to 2.0)"""

    top_p: float | None = None
    """Controls diversity via nucleus sampling (0.0 to 1.0)"""

    max_tokens: int | None = None
    """Maximum number of tokens to generate"""

    response_format: dict[str, Any] | type[BaseModel] | None = None
    """Format specification for the response"""

    stream: bool | None = None
    """Whether to stream the response"""

    n: int | None = None
    """Number of completions to generate"""

    stop: str | list[str] | None = None
    """Stop sequences for generation"""

    presence_penalty: float | None = None
    """Penalize new tokens based on presence in text"""

    frequency_penalty: float | None = None
    """Penalize new tokens based on frequency in text"""

    seed: int | None = None
    """Random seed for reproducible results"""

    user: str | None = None
    """Unique identifier for the end user"""

    parallel_tool_calls: bool | None = None
    """Whether to allow parallel tool calls"""

    logprobs: bool | None = None
    """Include token-level log probabilities in the response"""

    top_logprobs: int | None = None
    """Number of top alternatives to return when logprobs are requested"""

    logit_bias: dict[str, float] | None = None
    """Bias the likelihood of specified tokens during generation"""

    stream_options: dict[str, Any] | None = None
    """Additional options controlling streaming behavior"""

    max_completion_tokens: int | None = None
    """Maximum number of tokens for the completion (provider-dependent)"""

    reasoning_effort: Literal["minimal", "low", "medium", "high", "auto"] | None = "auto"
    """Reasoning effort level for models that support it. "auto" will map to each provider's default."""
