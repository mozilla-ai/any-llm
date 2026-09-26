"""Messages ↔ Responses conversion for OpenAI tools + thinking bridge."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from any_llm.tools import _flatten_responses_tool
from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJSONDelta,
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    MessageUsage,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
)
from any_llm.types.responses import ResponsesParams
from any_llm.utils.aio import aclose_quietly
from any_llm.utils.messages_compat import (
    _budget_to_reasoning_effort,
    _convert_message_to_openai,
    _convert_system_to_openai,
    _convert_tool_choice_to_openai,
    _convert_tools_to_openai,
)
from any_llm.utils.structured_output import is_structured_output_type, normalize_output_config

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from any_llm.types.messages import MessageContentBlock, MessageStreamEvent, MessagesParams, StopReason
    from any_llm.types.responses import ResponseStreamEvent


def messages_needs_responses(params: MessagesParams) -> bool:
    """True when tools + enabled thinking would be rejected on Chat Completions."""
    if not params.tools:
        return False
    thinking = params.thinking
    if not thinking:
        return False
    return thinking.get("type") == "enabled"


def messages_params_to_responses_params(params: MessagesParams) -> ResponsesParams:
    """Convert Anthropic-style MessagesParams to ResponsesParams."""
    instructions: str | None = None
    if params.system:
        instructions = _convert_system_to_openai(params.system)

    openai_messages: list[dict[str, Any]] = []
    for msg in params.messages:
        openai_messages.extend(_convert_message_to_openai(msg))
    input_items = _openai_messages_to_responses_input(openai_messages)

    result: dict[str, Any] = {
        "model": params.model,
        "input": input_items,
        "max_output_tokens": params.max_tokens,
    }
    if instructions is not None:
        result["instructions"] = instructions
    if params.temperature is not None:
        result["temperature"] = params.temperature
    if params.top_p is not None:
        result["top_p"] = params.top_p
    if params.stream is not None:
        result["stream"] = params.stream
    if params.prompt_cache_key is not None:
        result["prompt_cache_key"] = params.prompt_cache_key
    if params.service_tier is not None:
        result["service_tier"] = params.service_tier

    if params.tools:
        result["tools"] = [_flatten_responses_tool(tool) for tool in _convert_tools_to_openai(params.tools)]

    if params.tool_choice is not None:
        result["tool_choice"] = _tool_choice_to_responses(params.tool_choice)
        if params.tool_choice.get("disable_parallel_tool_use") is True:
            result["parallel_tool_calls"] = False

    if params.thinking and params.thinking.get("type") == "enabled":
        budget = params.thinking.get("budget_tokens", 8192)
        # Without summary=auto the API returns a reasoning item with an empty summary,
        # so response_to_message_response never emits a ThinkingBlock (#1432 QA).
        result["reasoning"] = {
            "effort": _budget_to_reasoning_effort(budget),
            "summary": "auto",
        }

    if params.output_format is not None:
        if is_structured_output_type(params.output_format):
            result["response_format"] = params.output_format
        else:
            fmt = normalize_output_config(params.output_format).get("format")
            if isinstance(fmt, dict) and fmt.get("schema"):
                schema = fmt["schema"]
                result["response_format"] = {
                    "type": "json_schema",
                    "name": schema.get("title", "structured_output"),
                    "schema": schema,
                }

    return ResponsesParams(**result)


def _tool_choice_to_responses(tool_choice: dict[str, Any]) -> str | dict[str, Any]:
    converted = _convert_tool_choice_to_openai(tool_choice)
    if isinstance(converted, dict) and converted.get("type") == "function" and "function" in converted:
        return {"type": "function", "name": converted["function"].get("name", "")}
    return converted


def _openai_messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content") or "",
                }
            )
            continue
        if role == "assistant":
            reasoning = msg.get("reasoning_content")
            if reasoning:
                items.append(
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": reasoning}],
                    }
                )
            content = msg.get("content")
            if content:
                items.append({"role": "assistant", "content": content})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue

        content = msg.get("content")
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for part in content:
                ptype = part.get("type")
                if ptype == "text":
                    parts.append({"type": "input_text", "text": part.get("text", "")})
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
                elif ptype == "file":
                    file_data = (part.get("file") or {}).get("file_data", "")
                    parts.append({"type": "input_file", "file_data": file_data})
                else:
                    parts.append(part)
            items.append({"role": role, "content": parts})
        else:
            items.append({"role": role, "content": content or ""})
    return items


def _item_attr(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def response_to_message_response(response: Any) -> MessageResponse:
    """Convert a Responses API result to an Anthropic MessageResponse."""
    content_blocks: list[MessageContentBlock] = []
    stop_reason: StopReason = "end_turn"

    for item in _item_attr(response, "output") or []:
        item_type = _item_attr(item, "type")
        if item_type == "reasoning":
            texts: list[str] = []
            for summary in _item_attr(item, "summary") or []:
                text = _item_attr(summary, "text")
                if text:
                    texts.append(text)
            for part in _item_attr(item, "content") or []:
                text = _item_attr(part, "text")
                if text:
                    texts.append(text)
            if texts:
                content_blocks.append(ThinkingBlock(type="thinking", thinking="".join(texts)))
        elif item_type == "message":
            for part in _item_attr(item, "content") or []:
                part_type = _item_attr(part, "type")
                if part_type == "output_text":
                    content_blocks.append(TextBlock(type="text", text=_item_attr(part, "text") or ""))
                elif part_type == "refusal":
                    content_blocks.append(TextBlock(type="text", text=_item_attr(part, "refusal") or ""))
                    stop_reason = "refusal"
        elif item_type == "function_call":
            args_raw = _item_attr(item, "arguments") or "{}"
            try:
                tool_input = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except (json.JSONDecodeError, TypeError):
                tool_input = {}
            content_blocks.append(
                ToolUseBlock(
                    type="tool_use",
                    id=_item_attr(item, "call_id") or _item_attr(item, "id") or "",
                    name=_item_attr(item, "name") or "",
                    input=tool_input if isinstance(tool_input, dict) else {},
                )
            )
            stop_reason = "tool_use"

    if not content_blocks:
        content_blocks.append(TextBlock(type="text", text=""))

    status = _item_attr(response, "status")
    if status == "incomplete" and stop_reason == "end_turn":
        stop_reason = "max_tokens"

    usage = MessageUsage(input_tokens=0, output_tokens=0)
    raw_usage = _item_attr(response, "usage")
    if raw_usage is not None:
        usage = MessageUsage(
            input_tokens=int(_item_attr(raw_usage, "input_tokens") or 0),
            output_tokens=int(_item_attr(raw_usage, "output_tokens") or 0),
        )

    return MessageResponse(
        id=_item_attr(response, "id") or "",
        type="message",
        role="assistant",
        content=content_blocks,
        model=_item_attr(response, "model") or "",
        stop_reason=stop_reason,
        usage=usage,
    )


class ResponsesStreamingState:
    """Tracks Messages stream lifecycle while consuming Responses events."""

    def __init__(self) -> None:
        self.started = False
        self.model = "unknown"
        self.message_id = ""
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason: StopReason | None = None
        self.block_by_item: dict[str, int] = {}
        self.open_indexes: set[int] = set()
        self.next_index = 0


def response_stream_event_to_message_events(
    event: Any,
    state: ResponsesStreamingState,
) -> list[MessageStreamEvent]:
    """Convert one Responses stream event into zero or more Messages events."""
    events: list[MessageStreamEvent] = []
    etype = _item_attr(event, "type")

    if etype == "response.created":
        response = _item_attr(event, "response")
        state.started = True
        state.message_id = _item_attr(response, "id") or ""
        state.model = _item_attr(response, "model") or state.model
        events.append(
            MessageStartEvent(
                type="message_start",
                message=MessageResponse(
                    id=state.message_id,
                    type="message",
                    role="assistant",
                    content=[],
                    model=state.model,
                    stop_reason=None,
                    usage=MessageUsage(input_tokens=0, output_tokens=0),
                ),
            )
        )
        return events

    if not state.started:
        state.started = True
        events.append(
            MessageStartEvent(
                type="message_start",
                message=MessageResponse(
                    id=state.message_id or "msg_stream",
                    type="message",
                    role="assistant",
                    content=[],
                    model=state.model,
                    stop_reason=None,
                    usage=MessageUsage(input_tokens=0, output_tokens=0),
                ),
            )
        )

    if etype == "response.output_item.added":
        item = _item_attr(event, "item")
        item_type = _item_attr(item, "type")
        item_id = _item_attr(item, "id") or f"item-{state.next_index}"
        index = state.next_index
        state.next_index += 1
        state.block_by_item[item_id] = index
        state.open_indexes.add(index)
        if item_type == "reasoning":
            events.append(
                ContentBlockStartEvent(
                    type="content_block_start",
                    index=index,
                    content_block=ThinkingBlock(type="thinking", thinking=""),
                )
            )
        elif item_type == "message":
            events.append(
                ContentBlockStartEvent(
                    type="content_block_start",
                    index=index,
                    content_block=TextBlock(type="text", text=""),
                )
            )
        elif item_type == "function_call":
            state.stop_reason = "tool_use"
            events.append(
                ContentBlockStartEvent(
                    type="content_block_start",
                    index=index,
                    content_block=ToolUseBlock(
                        type="tool_use",
                        id=_item_attr(item, "call_id") or "",
                        name=_item_attr(item, "name") or "",
                        input={},
                    ),
                )
            )
        return events

    if etype in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
        item_id = _item_attr(event, "item_id") or ""
        index = state.block_by_item.get(item_id)
        if index is None:
            return events
        delta_text = _item_attr(event, "delta") or ""
        if delta_text:
            events.append(
                ContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=index,
                    delta=ThinkingDelta(type="thinking_delta", thinking=delta_text),
                )
            )
        return events

    if etype == "response.output_text.delta":
        item_id = _item_attr(event, "item_id") or ""
        index = state.block_by_item.get(item_id)
        if index is None:
            return events
        delta_text = _item_attr(event, "delta") or ""
        if delta_text:
            events.append(
                ContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=index,
                    delta=TextDelta(type="text_delta", text=delta_text),
                )
            )
        return events

    if etype == "response.function_call_arguments.delta":
        item_id = _item_attr(event, "item_id") or ""
        index = state.block_by_item.get(item_id)
        if index is None:
            return events
        delta_text = _item_attr(event, "delta") or ""
        if delta_text:
            events.append(
                ContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=index,
                    delta=InputJSONDelta(type="input_json_delta", partial_json=delta_text),
                )
            )
        return events

    if etype == "response.output_item.done":
        item = _item_attr(event, "item")
        item_id = _item_attr(item, "id") or ""
        index = state.block_by_item.get(item_id)
        if index is not None and index in state.open_indexes:
            state.open_indexes.discard(index)
            events.append(ContentBlockStopEvent(type="content_block_stop", index=index))
        return events

    if etype in ("response.completed", "response.incomplete"):
        response = _item_attr(event, "response")
        raw_usage = _item_attr(response, "usage")
        if raw_usage is not None:
            state.input_tokens = int(_item_attr(raw_usage, "input_tokens") or 0)
            state.output_tokens = int(_item_attr(raw_usage, "output_tokens") or 0)
        if etype == "response.incomplete" and state.stop_reason is None:
            state.stop_reason = "max_tokens"
        for index in sorted(state.open_indexes):
            events.append(ContentBlockStopEvent(type="content_block_stop", index=index))
        state.open_indexes.clear()
        events.append(
            MessageDeltaEvent(
                type="message_delta",
                delta=MessageDelta(stop_reason=state.stop_reason or "end_turn"),
                usage=MessageDeltaUsage(
                    output_tokens=state.output_tokens,
                    input_tokens=state.input_tokens,
                ),
            )
        )
        events.append(MessageStopEvent(type="message_stop"))
        return events

    return events


async def convert_responses_stream(
    stream: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[MessageStreamEvent]:
    """Yield Messages stream events from a Responses event stream."""
    state = ResponsesStreamingState()
    try:
        async for event in stream:
            for msg_event in response_stream_event_to_message_events(event, state):
                yield msg_event
    finally:
        await aclose_quietly(stream)
