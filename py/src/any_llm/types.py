from collections.abc import Iterable
from typing import Any

from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import BaseModel

__all__ = [
    "ChatCompletion",
    "ChatCompletionAssistantMessageParam",
    "ChatCompletionChunk",
    "ChatCompletionContentPartImageParam",
    "ChatCompletionContentPartParam",
    "ChatCompletionContentPartTextParam",
    "ChatCompletionDeveloperMessageParam",
    "ChatCompletionFunctionMessageParam",
    "ChatCompletionMessage",
    "ChatCompletionMessageParam",
    "ChatCompletionMessageToolCall",
    "ChatCompletionMessageToolCallParam",
    "ChatCompletionSystemMessageParam",
    "ChatCompletionToolMessageParam",
    "ChatCompletionToolParam",
    "ChatCompletionUserMessageParam",
    "Choice",
    "ChoiceDelta",
    "ChunkChoice",
    "CompletionParams",
    "CompletionUsage",
]


class CompletionParams(BaseModel):
    """Parameters for a chat completion request.

    This provides a convenience interface for completion options that follows
    Python naming conventions while mapping to the OpenAI API parameters.
    """

    model: str
    messages: Iterable[ChatCompletionMessageParam]
    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    parallel_tool_calls: bool | None = None
    presence_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    tool_choice: str | dict[str, Any] | None = None
    tools: Iterable[ChatCompletionToolParam] | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    user: str | None = None

    model_config = {"extra": "allow"}

    def to_api_params(self) -> dict[str, Any]:
        """Convert to API parameters dictionary, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
