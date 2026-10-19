"""Assemble streamed chat completion chunks into a complete ``ChatCompletion``."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openai.types.chat.chat_completion import ChoiceLogprobs

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
    Choice,
    ChoiceDeltaToolCall,
    CompletionUsage,
    Function,
    ImageContent,
    Reasoning,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Iterable

    from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob


def _merge_extra_content(target: dict[str, Any], update: dict[str, Any]) -> None:
    """Merge ``update`` into ``target``; nested dicts merge, lists are joined, any other value is replaced.

    Providers send scalar ``extra_content`` values whole (a signature, an encrypted block), so a later
    chunk that repeats such a key replaces it. A list holds only the chunk's own items (Gemini's
    ``code_execution``), so lists are joined. Values are copied so the chunks are never modified.
    """
    for key, value in update.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge_extra_content(existing, value)
        elif isinstance(existing, list) and isinstance(value, list):
            existing.extend(copy.deepcopy(value))
        else:
            target[key] = copy.deepcopy(value)


@dataclass
class _ToolCallState:
    id: str | None = None
    name: str = ""
    arguments: str = ""
    extra_content: dict[str, Any] = field(default_factory=dict)

    def is_new_call(self, delta: ChoiceDeltaToolCall) -> bool:
        """Whether ``delta`` announces another call at the same ``index``.

        Several converters number calls within a chunk rather than across the stream, so two whole
        calls sent in separate chunks share an index; they still differ by id and both carry a name.
        """
        return bool(
            self.name
            and self.id is not None
            and delta.id
            and delta.id != self.id
            and delta.function is not None
            and delta.function.name
        )

    def add(self, delta: ChoiceDeltaToolCall) -> None:
        if delta.id and self.id is None:
            self.id = delta.id
        if delta.function is not None:
            # Some providers repeat the name on every chunk, others split it; keep a repeat out of the name.
            if delta.function.name and delta.function.name != self.name:
                self.name += delta.function.name
            if delta.function.arguments:
                self.arguments += delta.function.arguments
        if delta.extra_content:
            _merge_extra_content(self.extra_content, delta.extra_content)


@dataclass
class _ChoiceState:
    content: list[str] = field(default_factory=list)
    refusal: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: list[_ToolCallState] = field(default_factory=list)
    open_tool_calls: dict[int, _ToolCallState] = field(default_factory=dict)
    extra_content: dict[str, Any] = field(default_factory=dict)
    images: list[ImageContent] = field(default_factory=list)
    logprob_content: list[ChatCompletionTokenLogprob] = field(default_factory=list)
    logprob_refusal: list[ChatCompletionTokenLogprob] = field(default_factory=list)
    has_logprobs: bool = False
    finish_reason: str | None = None


class ChatCompletionAccumulator:
    """Build one ``ChatCompletion`` from the chunks of a streamed response.

    Feed every chunk to :meth:`add` in order, then read :attr:`completion`. The result carries what the
    non-streaming call would: text, refusal, reasoning, tool calls, the ``extra_content`` that providers
    use to carry state across turns (for example Gemini's ``thought_signature`` or Anthropic's thinking
    ``signature``), images, log probabilities and usage. Dumping ``completion.choices[0].message`` and
    appending it to ``messages`` replays the turn the same way as a non-streamed response.

    Tool-call fragments are grouped by their ``index`` and their ``arguments`` are concatenated; a
    fragment that carries a new id and a name at an index already in use starts another call.
    ``extra_content`` is merged key by key: lists are joined, and a later chunk replaces any other
    value of a repeated key. Audio output is not aggregated. The completion is built once and reused
    until another chunk is added.

    Example:
        >>> accumulator = ChatCompletionAccumulator()
        >>> for chunk in stream:
        ...     accumulator.add(chunk)
        >>> message = accumulator.completion.choices[0].message

    """

    def __init__(self) -> None:
        """Start with no chunks."""
        self._id: str | None = None
        self._model: str | None = None
        self._created: int | None = None
        self._system_fingerprint: str | None = None
        self._service_tier: str | None = None
        self._usage: CompletionUsage | None = None
        self._choices: dict[int, _ChoiceState] = {}
        self._completion: ChatCompletion | None = None

    def add(self, chunk: ChatCompletionChunk) -> None:
        """Fold one chunk into the running result."""
        self._completion = None
        self._id = self._id or chunk.id
        self._model = self._model or chunk.model
        self._created = self._created or chunk.created
        self._system_fingerprint = chunk.system_fingerprint or self._system_fingerprint
        self._service_tier = chunk.service_tier or self._service_tier
        if chunk.usage is not None:
            self._usage = chunk.usage

        for choice in chunk.choices:
            state = self._choices.setdefault(choice.index, _ChoiceState())
            delta = choice.delta
            if delta.content:
                state.content.append(delta.content)
            if delta.refusal:
                state.refusal.append(delta.refusal)
            if delta.reasoning is not None and delta.reasoning.content:
                state.reasoning.append(delta.reasoning.content)
            if delta.extra_content:
                _merge_extra_content(state.extra_content, delta.extra_content)
            if delta.images:
                state.images.extend(delta.images)
            for tool_call in delta.tool_calls or []:
                current = state.open_tool_calls.get(tool_call.index)
                if current is None or current.is_new_call(tool_call):
                    current = _ToolCallState()
                    state.tool_calls.append(current)
                    state.open_tool_calls[tool_call.index] = current
                current.add(tool_call)
            if choice.logprobs is not None:
                state.has_logprobs = True
                state.logprob_content.extend(choice.logprobs.content or [])
                state.logprob_refusal.extend(choice.logprobs.refusal or [])
            if choice.finish_reason is not None:
                state.finish_reason = choice.finish_reason

    @property
    def completion(self) -> ChatCompletion:
        """The completion assembled from the chunks added so far.

        Raises:
            ValueError: If no chunk has been added, or a choice has not reported a ``finish_reason`` yet,
                which means the stream stopped early.

        """
        if self._completion is not None:
            return self._completion
        if self._id is None or self._model is None or self._created is None:
            msg = "No chunks have been added; there is nothing to assemble."
            raise ValueError(msg)

        choices = []
        for index in sorted(self._choices):
            state = self._choices[index]
            if state.finish_reason is None:
                msg = f"Choice {index} has no finish_reason; the stream ended before it completed."
                raise ValueError(msg)
            choices.append(
                Choice(
                    index=index,
                    finish_reason=state.finish_reason,
                    message=self._build_message(state),
                    logprobs=(
                        ChoiceLogprobs(
                            content=state.logprob_content or None,
                            refusal=state.logprob_refusal or None,
                        )
                        if state.has_logprobs
                        else None
                    ),
                )
            )

        self._completion = ChatCompletion(
            id=self._id,
            object="chat.completion",
            created=self._created,
            model=self._model,
            choices=choices,
            usage=self._usage,
            system_fingerprint=self._system_fingerprint,
            service_tier=self._service_tier,
        )
        return self._completion

    @staticmethod
    def _build_message(state: _ChoiceState) -> ChatCompletionMessage:
        tool_calls: list[ChatCompletionMessageToolCall] = [
            ChatCompletionMessageFunctionToolCall(
                id=call.id or "",
                type="function",
                function=Function(name=call.name, arguments=call.arguments),
                extra_content=copy.deepcopy(call.extra_content) or None,
            )
            for call in state.tool_calls
        ]
        reasoning = "".join(state.reasoning)
        return ChatCompletionMessage(
            role="assistant",
            content="".join(state.content) or None,
            refusal="".join(state.refusal) or None,
            reasoning=Reasoning(content=reasoning) if reasoning else None,
            tool_calls=tool_calls or None,
            extra_content=copy.deepcopy(state.extra_content) or None,
            images=state.images or None,
        )


def accumulate_chunks(chunks: Iterable[ChatCompletionChunk]) -> ChatCompletion:
    """Consume a streamed response and return the assembled ``ChatCompletion``."""
    accumulator = ChatCompletionAccumulator()
    for chunk in chunks:
        accumulator.add(chunk)
    return accumulator.completion


async def aaccumulate_chunks(chunks: AsyncIterable[ChatCompletionChunk]) -> ChatCompletion:
    """Consume an async streamed response and return the assembled ``ChatCompletion``."""
    accumulator = ChatCompletionAccumulator()
    async for chunk in chunks:
        accumulator.add(chunk)
    return accumulator.completion
