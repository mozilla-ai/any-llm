"""Bidirectional conversion between Anthropic Messages API and OpenAI Chat Completions formats."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from any_llm.types.messages import (
    MessageContentBlock,
    MessageResponse,
    MessageStreamEvent,
    MessageUsage,
)

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
    from any_llm.types.messages import MessagesParams


def messages_params_to_completion_params(params: MessagesParams) -> dict[str, Any]:
    """Convert MessagesParams (Anthropic format) to kwargs suitable for CompletionParams.

    Returns a dict that can be passed to CompletionParams(**result).
    """
    messages: list[dict[str, Any]] = []

    if params.system:
        messages.append({"role": "system", "content": params.system})

    for msg in params.messages:
        converted = _convert_message_to_openai(msg)
        messages.extend(converted)

    result: dict[str, Any] = {
        "model_id": params.model,
        "messages": messages,
        "max_tokens": params.max_tokens,
    }

    if params.temperature is not None:
        result["temperature"] = params.temperature
    if params.top_p is not None:
        result["top_p"] = params.top_p
    if params.stop_sequences is not None:
        result["stop"] = params.stop_sequences
    if params.stream is not None:
        result["stream"] = params.stream

    if params.tools:
        result["tools"] = _convert_tools_to_openai(params.tools)

    if params.tool_choice is not None:
        result["tool_choice"] = _convert_tool_choice_to_openai(params.tool_choice)

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
    """Convert Anthropic assistant content blocks to OpenAI format."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
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
    return [result]


def _convert_user_blocks_to_openai(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic user content blocks to OpenAI format.

    Handles tool_result blocks (→ role:tool messages) and content blocks (text, image).
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
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                text_parts = [b.get("text", "") for b in tool_content if b.get("type") == "text"]
                tool_content = "".join(text_parts)
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": str(tool_content),
                }
            )
        elif block_type == "text":
            content_blocks.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
            else:
                url = source.get("url", "")
            content_blocks.append({"type": "image_url", "image_url": {"url": url}})
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


def chat_completion_to_message_response(completion: ChatCompletion) -> MessageResponse:
    """Convert an OpenAI ChatCompletion to an Anthropic MessageResponse."""
    content_blocks: list[MessageContentBlock] = []
    stop_reason = "end_turn"

    if completion.choices:
        choice = completion.choices[0]
        msg = choice.message

        if msg.reasoning:
            content_blocks.append(MessageContentBlock(type="thinking", thinking=msg.reasoning.content))

        if msg.content:
            content_blocks.append(MessageContentBlock(type="text", text=msg.content))

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
                    MessageContentBlock(
                        type="tool_use",
                        id=tc.id,
                        name=fn.name,
                        input=tool_input,
                    )
                )

        finish_reason = choice.finish_reason
        stop_reason = _finish_reason_to_stop_reason(finish_reason)

    if not content_blocks:
        content_blocks.append(MessageContentBlock(type="text", text=""))

    usage = MessageUsage(input_tokens=0, output_tokens=0)
    if completion.usage:
        usage = MessageUsage(
            input_tokens=completion.usage.prompt_tokens,
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


def _finish_reason_to_stop_reason(finish_reason: str | None) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
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
        self.tool_call_id: str | None = None
        self.tool_call_name: str | None = None


def chat_completion_chunk_to_message_stream_events(
    chunk: ChatCompletionChunk,
    state: StreamingState,
) -> list[MessageStreamEvent]:
    """Convert a ChatCompletionChunk to a list of MessageStreamEvents.

    This is stateful: it tracks the current content block index and type to emit
    the correct lifecycle events (start/delta/stop).
    """
    events: list[MessageStreamEvent] = []
    state.model = chunk.model

    if chunk.usage:
        if chunk.usage.prompt_tokens:
            state.input_tokens = chunk.usage.prompt_tokens
        if chunk.usage.completion_tokens:
            state.output_tokens = chunk.usage.completion_tokens

    if not state.started:
        state.started = True
        usage = MessageUsage(input_tokens=state.input_tokens, output_tokens=0)
        msg = MessageResponse(
            id=chunk.id,
            type="message",
            role="assistant",
            content=[],
            model=chunk.model,
            stop_reason=None,
            usage=usage,
        )
        events.append(MessageStreamEvent(type="message_start", message=msg))

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
                MessageStreamEvent(
                    type="content_block_start",
                    index=state.current_block_index,
                    content_block=MessageContentBlock(type="thinking", thinking=""),
                )
            )
        events.append(
            MessageStreamEvent(
                type="content_block_delta",
                index=state.current_block_index,
                delta={"type": "thinking_delta", "thinking": delta.reasoning.content},
            )
        )

    if delta.content is not None:
        if state.current_block_type != "text":
            _close_current_block(state, events)
            state.current_block_index += 1
            state.current_block_type = "text"
            events.append(
                MessageStreamEvent(
                    type="content_block_start",
                    index=state.current_block_index,
                    content_block=MessageContentBlock(type="text", text=""),
                )
            )
        if delta.content:
            events.append(
                MessageStreamEvent(
                    type="content_block_delta",
                    index=state.current_block_index,
                    delta={"type": "text_delta", "text": delta.content},
                )
            )

    if delta.tool_calls:
        for tc in delta.tool_calls:
            if tc.id:
                _close_current_block(state, events)
                state.current_block_index += 1
                state.current_block_type = "tool_use"
                state.tool_call_id = tc.id
                state.tool_call_name = tc.function.name if tc.function else ""
                events.append(
                    MessageStreamEvent(
                        type="content_block_start",
                        index=state.current_block_index,
                        content_block=MessageContentBlock(
                            type="tool_use",
                            id=state.tool_call_id,
                            name=state.tool_call_name,
                            input={},
                        ),
                    )
                )
            if tc.function and tc.function.arguments:
                events.append(
                    MessageStreamEvent(
                        type="content_block_delta",
                        index=state.current_block_index,
                        delta={"type": "input_json_delta", "partial_json": tc.function.arguments},
                    )
                )

    if choice.finish_reason:
        _close_current_block(state, events)
        stop_reason = _finish_reason_to_stop_reason(choice.finish_reason)
        events.append(
            MessageStreamEvent(
                type="message_delta",
                delta={"stop_reason": stop_reason},
                usage=MessageUsage(input_tokens=state.input_tokens, output_tokens=state.output_tokens),
            )
        )
        events.append(MessageStreamEvent(type="message_stop"))

    return events


def _close_current_block(state: StreamingState, events: list[MessageStreamEvent]) -> None:
    """Emit a content_block_stop event for the current block if one is open."""
    if state.current_block_type is not None:
        events.append(
            MessageStreamEvent(
                type="content_block_stop",
                index=state.current_block_index,
            )
        )
        state.current_block_type = None


def generate_message_id() -> str:
    """Generate a unique message ID in Anthropic format."""
    return f"msg_{uuid.uuid4().hex[:24]}"
