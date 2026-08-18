"""Bidirectional conversion between Anthropic Messages API and OpenAI Chat Completions formats."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from any_llm.exceptions import InvalidRequestError
from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJSONDelta,
    MessageResponse,
    MessageStartEvent,
    MessageUsage,
    StopReason,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
)
from any_llm.utils.structured_output import is_structured_output_type, normalize_output_config

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionUsage
    from any_llm.types.messages import MessageContentBlock, MessagesParams


def _output_config_to_response_format(output_config: dict[str, Any]) -> dict[str, Any]:
    """Translate a raw Anthropic ``output_config`` dict into a completion ``response_format``.

    Lets the bridge carry a non-Pydantic JSON schema to non-Anthropic providers: the schema
    under ``output_config["format"]["schema"]`` is rewrapped as the OpenAI ``json_schema``
    response format. The name falls back to the schema's ``title`` (or ``"structured_output"``).

    Both dict shapes ``normalize_output_config`` accepts are handled, and a dict carrying no
    schema raises there rather than putting a ``response_format`` on the wire that constrains
    nothing.
    """
    schema = normalize_output_config(output_config)["format"]["schema"]
    name = schema.get("title", "structured_output")
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


def _convert_system_to_openai(system: str | list[dict[str, Any]]) -> str:
    """Flatten an Anthropic system value to a plain string.

    Anthropic accepts a list of text blocks so callers can attach cache_control
    breakpoints. OpenAI-compatible backends validate the system message as
    str | list[content_part] and reject the extra cache_control key, so send
    the concatenated text instead.
    """
    if isinstance(system, str):
        return system
    return "".join(b.get("text", "") for b in system if b.get("type") == "text")


def messages_params_to_completion_params(params: MessagesParams) -> dict[str, Any]:
    """Convert MessagesParams (Anthropic format) to kwargs suitable for CompletionParams.

    Returns a dict that can be passed to CompletionParams(**result).
    """
    messages: list[dict[str, Any]] = []

    if params.system:
        messages.append({"role": "system", "content": _convert_system_to_openai(params.system)})

    for msg in params.messages:
        converted = _convert_message_to_openai(msg)
        messages.extend(converted)

    result: dict[str, Any] = {
        "model_id": params.model,
        "messages": messages,
        "max_tokens": params.max_tokens,
    }

    if params.prompt_cache_key is not None:
        result["prompt_cache_key"] = params.prompt_cache_key
    if params.service_tier is not None:
        result["service_tier"] = params.service_tier
    if params.temperature is not None:
        result["temperature"] = params.temperature
    if params.top_p is not None:
        result["top_p"] = params.top_p
    if params.stop_sequences is not None:
        result["stop"] = params.stop_sequences
    if params.stream is not None:
        result["stream"] = params.stream
        if params.stream:
            # OpenAI-compatible backends omit token usage from streamed chunks
            # unless asked for it, so the streamed Messages bridge would report
            # zero tokens. Request the trailing usage-only chunk that the
            # streaming wrapper flushes into the closing ``message_delta``.
            # Providers that don't support ``stream_options`` strip it in their
            # own param conversion, and the native Anthropic provider never
            # reaches this bridge (it overrides ``_amessages``).
            result["stream_options"] = {"include_usage": True}

    if params.output_format is not None:
        if is_structured_output_type(params.output_format):
            result["response_format"] = params.output_format
        else:
            result["response_format"] = _output_config_to_response_format(cast("dict[str, Any]", params.output_format))

    if params.tools:
        result["tools"] = _convert_tools_to_openai(params.tools)

    if params.tool_choice is not None:
        result["tool_choice"] = _convert_tool_choice_to_openai(params.tool_choice)
        # Anthropic carries the sequential-tool-use switch inside tool_choice; OpenAI carries it
        # as a sibling of it. Anthropic accepts the flag on every tool_choice type, so this does
        # not depend on which type _convert_tool_choice_to_openai resolved.
        if params.tool_choice.get("disable_parallel_tool_use") is True:
            result["parallel_tool_calls"] = False

    if params.thinking:
        if params.thinking.get("type") == "enabled":
            budget = params.thinking.get("budget_tokens", 8192)
            result["reasoning_effort"] = _budget_to_reasoning_effort(budget)
        elif params.thinking.get("type") == "disabled":
            result["reasoning_effort"] = "none"

    return result


def _convert_message_to_openai(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a single Anthropic-format message to one or more OpenAI-format messages."""
    role = msg.get("role", "user")
    content = msg.get("content")

    if isinstance(content, str):
        return [{"role": role, "content": content}]

    if not isinstance(content, list):
        return [{"role": role, "content": content}]

    if role == "assistant":
        return _convert_assistant_blocks_to_openai(content)

    if role == "user":
        return _convert_user_blocks_to_openai(content)

    return [{"role": role, "content": content}]


def _convert_assistant_blocks_to_openai(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic assistant content blocks to OpenAI format.

    A ``thinking`` block replayed from a previous turn becomes ``reasoning_content`` on the
    assistant message. That is the first entry in ``REASONING_FIELD_NAMES``, and the field
    ``deepseek``'s ``_reinject_reasoning_content`` restores for the same purpose, on the
    grounds it states there: of the reasoning fields any_llm knows about, ``reasoning_content``
    is the one that belongs on the wire. The normalized ``reasoning`` field would not do,
    because ``AnyLLM.acompletion`` strips it as an any_llm extension to the OpenAI spec.

    Dropping the block made the bridge lossy in one direction only:
    ``chat_completion_to_message_response`` builds a ``ThinkingBlock`` out of an upstream
    ``reasoning`` field, so a caller could read thinking out of a Messages response and then
    had nowhere to put it back on the next request.

    The Anthropic ``signature`` travels in the ``extra_content["anthropic"]`` side-channel that
    ``anthropic``'s ``_extract_anthropic_thinking_signature`` already reads, so a bridged
    request that later reaches an Anthropic-native provider can rebuild the block whole.
    Anthropic requires that signature back unmodified while extended thinking is on.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    signature: str | None = None
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "thinking":
            thinking_parts.append(block.get("thinking", ""))
            block_signature = block.get("signature")
            if isinstance(block_signature, str) and block_signature:
                signature = block_signature
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )

    result: dict[str, Any] = {"role": "assistant"}
    if text_parts:
        result["content"] = "".join(text_parts)
    else:
        result["content"] = None
    if tool_calls:
        result["tool_calls"] = tool_calls
    reasoning_content = "".join(thinking_parts)
    if reasoning_content:
        result["reasoning_content"] = reasoning_content
    if signature is not None:
        result["extra_content"] = {"anthropic": {"signature": signature}}
    return [result]


def _convert_image_block_to_openai(block: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic ``image`` block to an OpenAI ``image_url`` content part.

    Inverse of the ``image_url`` branch in ``anthropic``'s ``_convert_content_for_anthropic``.
    """
    source = block.get("source", {})
    if source.get("type") == "base64":
        url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    else:
        url = source.get("url", "")
    return {"type": "image_url", "image_url": {"url": url}}


def _convert_document_block_to_openai(block: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic ``document`` block to an OpenAI content part.

    Inverse of the ``file`` branch in ``anthropic``'s ``_convert_content_for_anthropic``, which
    pairs an Anthropic ``document`` with an OpenAI ``file`` part carrying a ``file_data`` data
    URI. Anthropic's document source is one of ``base64``, ``text``, ``content`` or ``url``.
    The two that already hold text (``text`` and ``content``) become a text part rather than a
    data URI wrapping plain text, since ``file`` has no equivalent for them.

    Raises:
        InvalidRequestError: when the source carries no payload. An empty ``file_data`` is not
            a usable attachment, so it would cost a backend round trip to learn the document
            never made it.

    """
    source = block.get("source", {})
    source_type = source.get("type")
    if source_type == "text":
        return {"type": "text", "text": source.get("data", "")}
    if source_type == "content":
        return {"type": "text", "text": _flatten_document_content_source(source.get("content"))}
    if source_type == "base64":
        media_type = source.get("media_type", "application/pdf")
        return {"type": "file", "file": {"file_data": f"data:{media_type};base64,{source.get('data', '')}"}}
    url = source.get("url", "")
    if not url:
        msg = f"document block source carries no payload (source type {source_type!r})"
        raise InvalidRequestError(msg)
    return {"type": "file", "file": {"file_data": url}}


def _flatten_document_content_source(content: Any) -> str:
    """Flatten a ``content``-source document into text.

    Anthropic's ``content`` source holds either a string or a list of text and image blocks.
    Only the text carries over: an OpenAI content part is a single typed value, so nested
    images cannot ride inside the text part this returns.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _convert_tool_result_content(tool_content: Any) -> tuple[str, list[dict[str, Any]]]:
    """Split an Anthropic ``tool_result`` payload into wire text and non-text content parts.

    OpenAI's ``role: tool`` message takes text only, so the two halves cannot ride in one
    message. Concatenating the text blocks and discarding the rest is what deleted image and
    document bytes that an agent had put in a tool result; the caller re-attaches the returned
    parts as a following ``user`` message instead.
    """
    if not isinstance(tool_content, list):
        return str(tool_content), []
    text_parts: list[str] = []
    extra_parts: list[dict[str, Any]] = []
    for block in tool_content:
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "image":
            extra_parts.append(_convert_image_block_to_openai(block))
        elif block_type == "document":
            extra_parts.append(_convert_document_block_to_openai(block))
    return "".join(text_parts), extra_parts


def _convert_user_blocks_to_openai(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic user content blocks to OpenAI format.

    Handles tool_result blocks (→ role:tool messages) and content blocks (text, image).

    A tool result marked ``is_error`` keeps that marker on the emitted ``role: tool`` message.
    OpenAI has no field for it, and the OpenAI SDK forwards unknown message keys verbatim, so
    the flag reaches a backend that understands it and is inert on one that does not. Dropping
    it left a failed tool call indistinguishable from a successful one whose output happened to
    read like an error.

    A tool result carrying image or document blocks emits the text as the ``role: tool``
    message and the remaining parts as a ``user`` message directly after it, because OpenAI
    accepts text only on a tool message. The bytes stay in the request at the same point in the
    conversation rather than being dropped.
    """
    results: list[dict[str, Any]] = []
    content_blocks: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type", "")
        if block_type == "tool_result":
            # Flush any accumulated content blocks first
            if content_blocks:
                results.append({"role": "user", "content": content_blocks})
                content_blocks = []
            tool_text, extra_parts = _convert_tool_result_content(block.get("content", ""))
            tool_message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": tool_text,
            }
            if block.get("is_error") is True:
                tool_message["is_error"] = True
            results.append(tool_message)
            if extra_parts:
                results.append({"role": "user", "content": extra_parts})
        elif block_type == "text":
            content_blocks.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "image":
            content_blocks.append(_convert_image_block_to_openai(block))
        else:
            content_blocks.append(block)

    if content_blocks:
        results.append({"role": "user", "content": content_blocks})

    return results


def _convert_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool format to OpenAI function tool format."""
    openai_tools = []
    for tool in tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return openai_tools


def _convert_tool_choice_to_openai(tool_choice: dict[str, Any]) -> str | dict[str, Any]:
    """Convert Anthropic tool_choice to OpenAI format."""
    tc_type = tool_choice.get("type", "auto")
    if tc_type == "auto":
        return "auto"
    if tc_type == "any":
        return "required"
    if tc_type == "none":
        return "none"
    if tc_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return "auto"


def _budget_to_reasoning_effort(budget: int) -> str:
    """Map thinking budget tokens to a reasoning_effort level."""
    if budget <= 1024:
        return "minimal"
    if budget <= 2048:
        return "low"
    if budget <= 8192:
        return "medium"
    if budget <= 24576:
        return "high"
    return "xhigh"


def split_cached_input_tokens(prompt_tokens: int, cached_tokens: int) -> tuple[int, int | None]:
    """Split an OpenAI prompt-token total into disjoint Anthropic input/cache-read counts.

    OpenAI reports ``prompt_tokens`` as the whole prompt with ``prompt_tokens_details.cached_tokens``
    as a subset of it, while Anthropic's ``input_tokens`` and ``cache_read_input_tokens`` are disjoint
    and sum to the prompt. Copying the cached count across without subtracting would make any consumer
    that sums the fields over-count, and would bill cached tokens twice in a cost model that prices
    the two at different rates.

    The cached count comes back as ``None`` rather than 0 when there was no cache hit, so a response
    from a provider that reports no cache accounting looks exactly as it did before this mapping
    existed. ``cache_creation_input_tokens`` is left unset rather than synthesized because no provider
    in this repo populates ``prompt_tokens_details.cache_write_tokens``, which is the field a cache
    write would arrive on.

    The cached count is clamped into ``[0, prompt_tokens]`` so a provider that reports the two
    inconsistently cannot push ``input_tokens`` negative (cached above the total) or above the prompt
    total (cached below zero). Clamping the subtrahend rather than flooring the result keeps the sum
    invariant intact: the two returned values still add up to ``prompt_tokens``.
    """
    cached = min(max(cached_tokens, 0), prompt_tokens)
    return prompt_tokens - cached, cached or None


def _cached_tokens_from_usage(usage: CompletionUsage) -> int:
    """Read ``prompt_tokens_details.cached_tokens`` off a usage object, defaulting to 0."""
    if usage.prompt_tokens_details is None:
        return 0
    return usage.prompt_tokens_details.cached_tokens or 0


def chat_completion_to_message_response(completion: ChatCompletion) -> MessageResponse:
    """Convert an OpenAI ChatCompletion to an Anthropic MessageResponse."""
    content_blocks: list[MessageContentBlock] = []
    stop_reason: StopReason = "end_turn"

    if completion.choices:
        choice = completion.choices[0]
        msg = choice.message

        if msg.reasoning:
            content_blocks.append(ThinkingBlock(type="thinking", thinking=msg.reasoning.content))

        if msg.content:
            content_blocks.append(TextBlock(type="text", text=msg.content))

        if msg.tool_calls:
            for tc in msg.tool_calls:
                if not hasattr(tc, "function"):
                    continue
                fn = tc.function
                try:
                    tool_input = json.loads(fn.arguments) if fn.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    tool_input = {}
                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=tc.id,
                        name=fn.name,
                        input=tool_input,
                    )
                )

        finish_reason = choice.finish_reason
        stop_reason = _finish_reason_to_stop_reason(finish_reason)

    if not content_blocks:
        content_blocks.append(TextBlock(type="text", text=""))

    usage = MessageUsage(input_tokens=0, output_tokens=0)
    if completion.usage:
        input_tokens, cache_read = split_cached_input_tokens(
            completion.usage.prompt_tokens,
            _cached_tokens_from_usage(completion.usage),
        )
        usage = MessageUsage(
            input_tokens=input_tokens,
            cache_read_input_tokens=cache_read,
            output_tokens=completion.usage.completion_tokens,
        )

    return MessageResponse(
        id=completion.id,
        type="message",
        role="assistant",
        content=content_blocks,
        model=completion.model,
        stop_reason=stop_reason,
        usage=usage,
    )


def _finish_reason_to_stop_reason(finish_reason: str | None) -> StopReason:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    mapping: dict[str, StopReason] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
        "function_call": "tool_use",
    }
    return mapping.get(finish_reason or "stop", "end_turn")


class StreamingState:
    """Tracks state during streaming conversion from ChatCompletionChunks to MessageStreamEvents."""

    def __init__(self) -> None:
        """Initialize streaming state."""
        self.started = False
        self.current_block_index = -1
        self.current_block_type: str | None = None
        self.model = "unknown"
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_input_tokens = 0
        self.stop_reason: StopReason | None = None
        self.tool_call_id: str | None = None
        self.tool_call_name: str | None = None
        self.tool_block_indexes: dict[int, int] = {}
        """Content block index of each open tool_use block, keyed by OpenAI ``tool_calls[].index``."""


def chat_completion_chunk_to_message_stream_events(
    chunk: ChatCompletionChunk,
    state: StreamingState,
) -> list[MessageStartEvent | ContentBlockStartEvent | ContentBlockDeltaEvent | ContentBlockStopEvent]:
    """Convert a ChatCompletionChunk to a list of MessageStreamEvents.

    This is stateful: it tracks the current content block index and type to emit
    the correct lifecycle events (start/delta/stop).
    """
    events: list[MessageStartEvent | ContentBlockStartEvent | ContentBlockDeltaEvent | ContentBlockStopEvent] = []
    state.model = chunk.model

    if chunk.usage:
        if chunk.usage.prompt_tokens:
            state.input_tokens = chunk.usage.prompt_tokens
        if chunk.usage.completion_tokens:
            state.output_tokens = chunk.usage.completion_tokens
        cached = _cached_tokens_from_usage(chunk.usage)
        if cached:
            state.cache_read_input_tokens = cached

    if not state.started:
        state.started = True
        input_tokens, cache_read = split_cached_input_tokens(state.input_tokens, state.cache_read_input_tokens)
        usage = MessageUsage(
            input_tokens=input_tokens,
            cache_read_input_tokens=cache_read,
            output_tokens=0,
        )
        msg = MessageResponse(
            id=chunk.id,
            type="message",
            role="assistant",
            content=[],
            model=chunk.model,
            stop_reason=None,
            usage=usage,
        )
        events.append(MessageStartEvent(type="message_start", message=msg))

    if not chunk.choices:
        return events

    choice = chunk.choices[0]
    delta = choice.delta

    if delta.reasoning and delta.reasoning.content is not None:
        if state.current_block_type != "thinking":
            _close_current_block(state, events)
            state.current_block_index += 1
            state.current_block_type = "thinking"
            events.append(
                ContentBlockStartEvent(
                    type="content_block_start",
                    index=state.current_block_index,
                    content_block=ThinkingBlock(type="thinking", thinking=""),
                )
            )
        events.append(
            ContentBlockDeltaEvent(
                type="content_block_delta",
                index=state.current_block_index,
                delta=ThinkingDelta(type="thinking_delta", thinking=delta.reasoning.content),
            )
        )

    if delta.content is not None:
        if state.current_block_type != "text":
            _close_current_block(state, events)
            state.current_block_index += 1
            state.current_block_type = "text"
            events.append(
                ContentBlockStartEvent(
                    type="content_block_start",
                    index=state.current_block_index,
                    content_block=TextBlock(type="text", text=""),
                )
            )
        if delta.content:
            events.append(
                ContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=state.current_block_index,
                    delta=TextDelta(type="text_delta", text=delta.content),
                )
            )

    if delta.tool_calls:
        for tc in delta.tool_calls:
            # An id repeated on later fragments of the same tool call must not open a second block.
            if tc.id and tc.index not in state.tool_block_indexes:
                # Parallel tool calls share one tool_use section: only a text or thinking block
                # is closed here, so a block stays open for every call still receiving arguments.
                if state.current_block_type != "tool_use":
                    _close_current_block(state, events)
                state.current_block_index += 1
                state.current_block_type = "tool_use"
                state.tool_call_id = tc.id
                state.tool_call_name = tc.function.name if tc.function else ""
                state.tool_block_indexes[tc.index] = state.current_block_index
                events.append(
                    ContentBlockStartEvent(
                        type="content_block_start",
                        index=state.current_block_index,
                        content_block=ToolUseBlock(
                            type="tool_use",
                            id=state.tool_call_id or "",
                            name=state.tool_call_name or "",
                            input={},
                        ),
                    )
                )
            if tc.function and tc.function.arguments:
                # Providers may interleave the fragments of parallel calls, so the destination
                # block comes from the tool call's own index rather than from the newest block.
                events.append(
                    ContentBlockDeltaEvent(
                        type="content_block_delta",
                        index=state.tool_block_indexes.get(tc.index, state.current_block_index),
                        delta=InputJSONDelta(type="input_json_delta", partial_json=tc.function.arguments),
                    )
                )

    if choice.finish_reason:
        _close_current_block(state, events)
        state.stop_reason = _finish_reason_to_stop_reason(choice.finish_reason)

    return events


def close_open_blocks(state: StreamingState) -> list[ContentBlockStopEvent]:
    """Build a content_block_stop event for every block still open, in block order.

    A tool_use section can hold more than one open block, because each parallel tool call gets
    its own block and stays open until the section ends.
    """
    if state.current_block_type is None:
        return []
    open_indexes = sorted(state.tool_block_indexes.values()) or [state.current_block_index]
    state.tool_block_indexes.clear()
    state.current_block_type = None
    return [ContentBlockStopEvent(type="content_block_stop", index=index) for index in open_indexes]


def _close_current_block(
    state: StreamingState,
    events: list[MessageStartEvent | ContentBlockStartEvent | ContentBlockDeltaEvent | ContentBlockStopEvent],
) -> None:
    """Emit content_block_stop events for any open blocks."""
    events.extend(close_open_blocks(state))
