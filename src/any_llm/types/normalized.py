from typing import Literal, NotRequired, Sequence, TypedDict


class NormalizedFunctionArgs(TypedDict):
    name: str
    arguments: str


class NormalizedToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: NormalizedFunctionArgs


class NormalizedMessage(TypedDict, total=False):
    role: Literal["developer", "system", "user", "assistant", "tool"]
    content: str | None
    tool_calls: NotRequired[Sequence[NormalizedToolCall] | None]
    reasoning_content: NotRequired[str | None]


class NormalizedChoice(TypedDict):
    message: NormalizedMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    index: int


class NormalizedUsage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class NormalizedResponse(TypedDict):
    id: str
    model: str
    created: int
    choices: Sequence[NormalizedChoice]
    usage: NotRequired[NormalizedUsage | None]

