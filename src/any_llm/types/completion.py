from typing import Any, Generic, Literal, TypeVar

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
from openai.types.completion_usage import CompletionTokensDetails as OpenAICompletionTokensDetails
from openai.types.completion_usage import CompletionUsage as OpenAICompletionUsage
from openai.types.completion_usage import PromptTokensDetails as OpenAIPromptTokensDetails
from openai.types.create_embedding_response import Usage as OpenAIUsage
from openai.types.embedding import Embedding as OpenAIEmbedding
from pydantic import BaseModel, ConfigDict, field_validator, model_serializer, model_validator

# See https://github.com/mozilla-ai/any-llm/issues/95:
# OpenAI Completion API doesn't include reasoning information, so we need to extend the openai type


class Reasoning(BaseModel):
    """Reasoning content emitted by a model.

    Serializes as a plain JSON string so that responses are compatible with
    OpenAI-style clients that expect ``delta.reasoning`` / ``message.reasoning``
    to be a string. The Python attribute ``content`` remains available for
    typed access (e.g. ``message.reasoning.content``).
    """

    content: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_input(cls, value: Any) -> Any:
        """Accept either a plain string or the ``{"content": str}`` object form."""
        if isinstance(value, str):
            return {"content": value}
        if isinstance(value, dict) and "content" in value and value["content"] is not None:
            return {"content": str(value["content"])}
        return value

    @model_serializer
    def _serialize(self) -> str:
        """Serialize as a plain string for OpenAI-compatible wire format."""
        return self.content


class ChatCompletionMessageFunctionToolCall(OpenAIChatCompletionMessageFunctionToolCall):
    """Extended tool call type that includes extra_content for provider-specific data.

    The extra_content field is used to store provider-specific metadata that needs
    to be preserved across multi-turn conversations. For example, Gemini 3 models
    require thought_signature to be passed back with function calls.

    Example extra_content structure for Gemini:
        {"google": {"thought_signature": "<base64-encoded-signature>"}}
    """

    extra_content: dict[str, Any] | None = None


ChatCompletionMessageToolCall = ChatCompletionMessageFunctionToolCall | OpenAIChatCompletionMessageToolCall


class ChatCompletionMessage(OpenAIChatCompletionMessage):
    tool_calls: list[ChatCompletionMessageToolCall] | None = None  # type: ignore[assignment]
    reasoning: Reasoning | None = None
    annotations: list[dict[str, Any]] | None = None  # type: ignore[assignment]
    extra_content: dict[str, Any] | None = None
    """Provider-specific metadata that needs to be preserved across multi-turn conversations.

    For example, Anthropic's extended thinking requires the encrypted ``signature`` of a
    ``thinking`` block to be passed back unmodified alongside subsequent tool calls.

    Example extra_content structure for Anthropic:
        {"anthropic": {"signature": "<encrypted-signature>"}}
    """


class Choice(OpenAIChoice):
    message: ChatCompletionMessage


class ChatCompletion(OpenAIChatCompletion):
    choices: list[Choice]  # type: ignore[assignment]
    service_tier: str | None = None  # type: ignore[assignment]


ContentType = TypeVar("ContentType")


class ParsedChatCompletionMessage(ChatCompletionMessage, Generic[ContentType]):
    parsed: ContentType | None = None


class ParsedChoice(Choice, Generic[ContentType]):
    message: ParsedChatCompletionMessage[ContentType]


class ParsedChatCompletion(ChatCompletion, Generic[ContentType]):
    choices: list[ParsedChoice[ContentType]]  # type: ignore[assignment]


class ChoiceDeltaToolCall(OpenAIChoiceDeltaToolCall):
    """Streaming counterpart of ``ChatCompletionMessageFunctionToolCall``.

    Adds the same ``extra_content`` field so provider-specific tool-call metadata (e.g.
    Gemini's ``thought_signature``) can be carried on streaming deltas, not just on the
    final non-streaming tool call.
    """

    extra_content: dict[str, Any] | None = None


class ChoiceDelta(OpenAIChoiceDelta):
    reasoning: Reasoning | None = None
    tool_calls: list[ChoiceDeltaToolCall] | None = None  # type: ignore[assignment]
    extra_content: dict[str, Any] | None = None
    """Streaming counterpart of ``ChatCompletionMessage.extra_content``.

    Carries provider-specific metadata (e.g. Anthropic's thinking block ``signature``)
    that arrives as part of a streaming delta rather than the final message.
    """


class ChunkChoice(OpenAIChunkChoice):
    delta: ChoiceDelta


class ChatCompletionChunk(OpenAIChatCompletionChunk):
    choices: list[ChunkChoice]  # type: ignore[assignment]
    service_tier: str | None = None  # type: ignore[assignment]


Function = OpenAIFunction


class CacheUsageDetails(BaseModel):
    """Provider-neutral cache meters preserved alongside completion usage."""

    read_input_tokens: int | None = None
    creation_input_tokens: int | None = None
    creation_5m_input_tokens: int | None = None
    creation_1h_input_tokens: int | None = None
    included_in_prompt_tokens: bool | None = None
    provider_meters: dict[str, int] | None = None


class CompletionUsage(OpenAICompletionUsage):
    """OpenAI-compatible usage with optional provider cache accounting."""

    cache_usage: CacheUsageDetails | None = None

    @model_validator(mode="before")
    @classmethod
    def _preserve_cache_usage(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if value.get("cache_usage") is not None:
            return value

        def first(*names: str) -> int | None:
            for name in names:
                if value.get(name) is not None:
                    return value[name]
            return None

        read = first("cache_read_input_tokens", "prompt_cache_hit_tokens")
        creation = first("cache_creation_input_tokens", "prompt_cache_write_tokens")
        details = value.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
        else:
            cached = getattr(details, "cached_tokens", None)
        if read is None and cached is not None:
            read = cached

        ttl = value.get("cache_creation") or {}
        if not isinstance(ttl, dict):
            ttl = {}
        meters = {
            key: meter
            for key, meter in value.items()
            if key
            not in {
                "cache_read_input_tokens",
                "prompt_cache_hit_tokens",
                "cache_creation_input_tokens",
                "prompt_cache_write_tokens",
                "cache_creation",
                "prompt_tokens_details",
                "cache_usage",
            }
            and key.startswith(("prompt_cache_", "cache_"))
            and isinstance(meter, int)
        }
        if read is None and creation is None and not ttl and not meters:
            return value
        value = dict(value)
        value["cache_usage"] = {
            "read_input_tokens": read,
            "creation_input_tokens": creation,
            "creation_5m_input_tokens": ttl.get("ephemeral_5m_input_tokens"),
            "creation_1h_input_tokens": ttl.get("ephemeral_1h_input_tokens"),
            "included_in_prompt_tokens": "prompt_cache_hit_tokens" in value or cached is not None,
            "provider_meters": meters or None,
        }
        return value
CompletionTokensDetails = OpenAICompletionTokensDetails
PromptTokensDetails = OpenAIPromptTokensDetails
CreateEmbeddingResponse = OpenAICreateEmbeddingResponse
Embedding = OpenAIEmbedding
Usage = OpenAIUsage
ChoiceDeltaToolCallFunction = OpenAIChoiceDeltaToolCallFunction

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "auto"]


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

    tools: list[dict[str, Any] | Any] | None = None
    """List of tools for tool calling. Should be converted to OpenAI tool format dicts"""

    tool_choice: str | dict[str, Any] | None = None
    """Controls which tools the model can call"""

    temperature: float | None = None
    """Controls randomness in the response (0.0 to 2.0)"""

    top_p: float | None = None
    """Controls diversity via nucleus sampling (0.0 to 1.0)"""

    max_tokens: int | None = None
    """Maximum number of tokens to generate"""

    response_format: dict[str, Any] | type | None = None
    """Format specification for the response. Accepts Pydantic BaseModel subclasses, dataclass types, or dicts."""

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

    reasoning_effort: ReasoningEffort | None = "auto"
    """Reasoning effort level for models that support it. "auto" will map to each provider's default."""

    prompt_cache_key: str | None = None
    """A key to use when reading from or writing to a provider's prompt cache."""

    service_tier: str | None = None
    """The service tier to use for this request."""
