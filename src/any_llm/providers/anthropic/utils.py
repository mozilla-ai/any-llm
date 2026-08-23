import json
from datetime import datetime
from typing import Any, cast

from anthropic import transform_schema
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    Message,
    MessageDeltaEvent,
    MessageStopEvent,
)
from anthropic.types.model_info import ModelInfo as AnthropicModelInfo
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.logging import logger
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
    Choice,
    CompletionParams,
    CompletionUsage,
    Function,
    PromptTokensDetails,
    Reasoning,
)
from any_llm.types.model import Model
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type

DEFAULT_MAX_TOKENS = 8192
_ANTHROPIC_CONTENT_FILTER_REFUSAL = "Response blocked by Anthropic content filtering."
# OpenAI has no counterpart for the "stop_sequence" and "pause_turn" stop reasons, so those
# fall through to the "stop" default. "refusal" (a safety stop) and
# "model_context_window_exceeded" (the model ran out of context rather than out of max_tokens)
# do have one, and without it a refused or truncated answer looks like a normal completion.
# See https://docs.claude.com/en/docs/build-with-claude/handling-stop-reasons
ANTHROPIC_STOP_REASON_TO_FINISH_REASON = {
    "end_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}
REASONING_EFFORT_TO_ANTHROPIC_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


def _refusal_stop_details(value: object) -> dict[str, Any] | None:
    """Return typed Anthropic refusal details when the installed SDK exposes them."""
    stop_details = getattr(value, "stop_details", None)
    if isinstance(stop_details, BaseModel):
        return stop_details.model_dump(mode="json", exclude_none=True)
    return None


def _is_tool_call(message: dict[str, Any]) -> bool:
    """Check if the message is a tool call message."""
    return message["role"] == "assistant" and message.get("tool_calls") is not None


def _extract_anthropic_thinking_signature(message: dict[str, Any]) -> str | None:
    """Extract the encrypted thinking signature stored on a message's extra_content, if any."""
    extra_content = message.get("extra_content")
    if isinstance(extra_content, dict) and isinstance(anthropic_extra := extra_content.get("anthropic"), dict):
        signature = anthropic_extra.get("signature")
        if isinstance(signature, str):
            return signature
    return None


def _extract_reasoning_text(message: dict[str, Any]) -> str:
    """Extract the plain-text reasoning content from a message, regardless of its shape.

    ``reasoning`` may be a plain string (the OpenAI-wire-compatible serialized form) or a
    ``{"content": str}`` dict, depending on how the caller constructed the message.
    """
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, dict) and isinstance(content := reasoning.get("content"), str):
        return content
    return ""


def _build_anthropic_thinking_block(message: dict[str, Any]) -> dict[str, Any] | None:
    """Reconstruct an Anthropic ``thinking`` content block for replay across turns.

    When extended thinking is enabled, Anthropic requires the ``thinking`` block (including
    its encrypted ``signature``) to be passed back unmodified on subsequent turns, e.g.
    alongside tool results. Without it, the model loses its original reasoning trace, which
    can lead to degraded or repeated reasoning. See
    https://docs.claude.com/en/docs/build-with-claude/extended-thinking#preserving-thinking-blocks
    """
    signature = _extract_anthropic_thinking_signature(message)
    if signature is None:
        return None
    return {"type": "thinking", "thinking": _extract_reasoning_text(message), "signature": signature}


def _convert_content_for_anthropic(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert content blocks from OpenAI format to Anthropic format.
    - Parse the "content" field block by block
    - Convert image_url blocks to Anthropic image format
    - Convert file blocks (PDFs) to Anthropic document format
    """
    converted_content = []
    for block in content:
        if block.get("type") == "image_url":
            converted_block: dict[str, Any] = {"type": "image"}
            url = block.get("image_url", {}).get("url", "")
            if url[:5] == "data:":
                mime_part = url[5:]
                semi_idx = mime_part.find(";")
                media_type = mime_part[:semi_idx] if semi_idx != -1 else mime_part
                converted_block["source"] = {
                    "type": "base64",
                    "media_type": media_type,
                    "data": url.split("base64,")[1],
                }
            else:
                converted_block["source"] = {"type": "url", "url": url}
            converted_content.append(converted_block)
        elif block.get("type") == "file":
            file_data = block.get("file", {}).get("file_data", "")
            converted_block = {"type": "document"}
            if file_data[:5] == "data:":
                mime_part = file_data[5:]
                semi_idx = mime_part.find(";")
                media_type = mime_part[:semi_idx] if semi_idx != -1 else mime_part
                converted_block["source"] = {
                    "type": "base64",
                    "media_type": media_type,
                    "data": file_data.split("base64,")[1],
                }
            else:
                converted_block["source"] = {"type": "url", "url": file_data}
            converted_content.append(converted_block)
        else:
            converted_content.append(block)
    return converted_content


def _convert_messages_for_anthropic(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert messages to Anthropic format.

    - Extract messages with `role=system`.
    - Replace `role=tool` with `role=user`, according to examples in https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/.
    - Handle multiple tool calls in a single assistant message.
    - Merge consecutive tool results into a single user message.
    """
    system_message = None
    filtered_messages: list[dict[str, Any]] = []

    for message in messages:
        if message["role"] == "system":
            if system_message is None:
                system_message = message["content"]
            else:
                system_message += "\n" + message["content"]
        else:
            # Handle messages inside agent loop.
            # See https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview#tool-use-examples
            if _is_tool_call(message):
                # Convert ALL tool calls from the assistant message
                content_blocks: list[dict[str, Any]] = []
                if thinking_block := _build_anthropic_thinking_block(message):
                    content_blocks.append(thinking_block)
                for tool_call in message["tool_calls"]:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"]),
                        }
                    )
                message = {
                    "role": "assistant",
                    "content": content_blocks,
                }
            elif message["role"] == "tool":
                # Use tool_call_id from the message itself
                tool_use_id = message.get("tool_call_id", "")
                tool_result = {"type": "tool_result", "tool_use_id": tool_use_id, "content": message["content"]}

                # Check if the previous message is already a user message with tool_results
                # If so, merge this tool_result into it
                if (
                    filtered_messages
                    and filtered_messages[-1]["role"] == "user"
                    and isinstance(filtered_messages[-1]["content"], list)
                    and filtered_messages[-1]["content"]
                    and filtered_messages[-1]["content"][0].get("type") == "tool_result"
                ):
                    filtered_messages[-1]["content"].append(tool_result)
                    continue

                message = {
                    "role": "user",
                    "content": [tool_result],
                }
            elif message["role"] == "assistant" and (thinking_block := _build_anthropic_thinking_block(message)):
                # existing_content may be None (a reasoning-only turn with no text/tool_calls),
                # a plain string, or a list of content blocks.
                existing_content = message.get("content")
                content_blocks = [thinking_block]
                if isinstance(existing_content, str):
                    if existing_content:
                        content_blocks.append({"type": "text", "text": existing_content})
                elif isinstance(existing_content, list):
                    content_blocks.extend(existing_content)
                message = {
                    "role": "assistant",
                    "content": content_blocks,
                }

            if "content" in message and isinstance(message["content"], list):
                message["content"] = _convert_content_for_anthropic(message["content"])

            # Only keep Anthropic-compatible fields (strips OpenAI-specific fields like 'refusal')
            filtered_messages.append({"role": message["role"], "content": message.get("content", "")})

    return system_message, filtered_messages


def _create_openai_chunk_from_anthropic_chunk(chunk: Any, model_id: str) -> ChatCompletionChunk:
    """Convert Anthropic streaming chunk to OpenAI ChatCompletionChunk format."""
    chunk_dict = {
        "id": f"chatcmpl-{hash(str(chunk))}",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model_id,
        "choices": [],
        "usage": None,
    }

    delta: dict[str, Any] = {}
    finish_reason = None

    if isinstance(chunk, ContentBlockStartEvent):
        if chunk.content_block.type == "text":
            delta = {"content": ""}
        elif chunk.content_block.type == "tool_use":
            delta = {
                "tool_calls": [
                    {
                        "index": chunk.index,
                        "id": chunk.content_block.id,
                        "type": "function",
                        "function": {"name": chunk.content_block.name, "arguments": ""},
                    }
                ]
            }
        elif chunk.content_block.type == "thinking":
            delta = {"reasoning": {"content": ""}}

    elif isinstance(chunk, ContentBlockDeltaEvent):
        if chunk.delta.type == "text_delta":
            delta = {"content": chunk.delta.text}
        elif chunk.delta.type == "input_json_delta":
            delta = {
                "tool_calls": [
                    {
                        "index": chunk.index,
                        "function": {"arguments": chunk.delta.partial_json},
                    }
                ]
            }
        elif chunk.delta.type == "thinking_delta":
            delta = {"reasoning": {"content": chunk.delta.thinking}}
        elif chunk.delta.type == "signature_delta":
            # The encrypted signature of the thinking block. Must be preserved unmodified
            # and passed back to Anthropic on subsequent turns (e.g. alongside tool results)
            # to maintain reasoning continuity. See https://docs.claude.com/en/docs/build-with-claude/extended-thinking
            delta = {"extra_content": {"anthropic": {"signature": chunk.delta.signature}}}

    elif isinstance(chunk, ContentBlockStopEvent):
        finish_reason = None

    elif isinstance(chunk, MessageDeltaEvent):
        stop_reason = chunk.delta.stop_reason
        finish_reason = (
            ANTHROPIC_STOP_REASON_TO_FINISH_REASON.get(stop_reason, "stop") if stop_reason is not None else None
        )
        if finish_reason == "content_filter":
            delta = {"refusal": _ANTHROPIC_CONTENT_FILTER_REFUSAL}
        if stop_details := _refusal_stop_details(chunk.delta):
            delta["extra_content"] = {"anthropic": {"stop_details": stop_details}}

    elif isinstance(chunk, MessageStopEvent):
        finish_reason = None
        if hasattr(chunk, "message") and chunk.message.usage:
            anthropic_usage = chunk.message.usage
            cache_read = anthropic_usage.cache_read_input_tokens or 0
            cache_creation = anthropic_usage.cache_creation_input_tokens or 0
            total_prompt_tokens = anthropic_usage.input_tokens + cache_read + cache_creation
            chunk_dict["usage"] = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": anthropic_usage.output_tokens,
                "total_tokens": total_prompt_tokens + anthropic_usage.output_tokens,
                "prompt_tokens_details": PromptTokensDetails(cached_tokens=cache_read) if cache_read else None,
                "cache_creation_input_tokens": anthropic_usage.cache_creation_input_tokens,
                "cache_creation": anthropic_usage.cache_creation,
            }

    choice = {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
        "logprobs": None,
    }

    chunk_dict["choices"] = [choice]

    return ChatCompletionChunk.model_validate(chunk_dict)


def _convert_response(response: Message) -> ChatCompletion:
    """Convert Anthropic Message to OpenAI ChatCompletion format."""
    finish_reason_raw = response.stop_reason or "end_turn"
    finish_reason = ANTHROPIC_STOP_REASON_TO_FINISH_REASON.get(finish_reason_raw, "stop")

    content_parts: list[str] = []
    tool_calls: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] = []
    reasoning_content: str | None = None
    thinking_signature: str | None = None
    for content_block in response.content:
        if content_block.type == "text":
            content_parts.append(content_block.text)
        elif content_block.type == "tool_use":
            tool_calls.append(
                ChatCompletionMessageFunctionToolCall(
                    id=content_block.id,
                    type="function",
                    function=Function(
                        name=content_block.name,
                        arguments=json.dumps(content_block.input),
                    ),
                )
            )
        elif content_block.type == "thinking":
            if reasoning_content is None:
                reasoning_content = content_block.thinking
            else:
                reasoning_content += content_block.thinking
            # The encrypted signature must be preserved and replayed unmodified on
            # subsequent turns (e.g. alongside tool results) to maintain reasoning
            # continuity. See https://docs.claude.com/en/docs/build-with-claude/extended-thinking
            if content_block.signature:
                thinking_signature = content_block.signature
        elif content_block.type == "redacted_thinking":
            # Anthropic encrypts thinking that its safety systems flag, so the block carries
            # no readable text to surface. The rest of the turn is a normal response.
            logger.debug("Skipping redacted_thinking block with no readable content.")
        else:
            # Server-side tool blocks (web search, code execution, ...) have no Chat
            # Completions equivalent. Dropping them keeps the answer the model did return,
            # which is what the streaming converter already does for the same block types.
            logger.warning("Skipping unsupported Anthropic content block type: %s", content_block.type)

    anthropic_extra_content: dict[str, Any] = {}
    if thinking_signature:
        anthropic_extra_content["signature"] = thinking_signature
    if stop_details := _refusal_stop_details(response):
        anthropic_extra_content["stop_details"] = stop_details

    message = ChatCompletionMessage(
        role="assistant",
        content="".join(content_parts),
        refusal=_ANTHROPIC_CONTENT_FILTER_REFUSAL if finish_reason == "content_filter" else None,
        reasoning=Reasoning(content=reasoning_content) if reasoning_content else None,
        tool_calls=tool_calls or None,
        extra_content={"anthropic": anthropic_extra_content} if anthropic_extra_content else None,
    )

    cache_read = response.usage.cache_read_input_tokens or 0
    cache_creation_value = response.usage.cache_creation_input_tokens
    cache_creation = cache_creation_value or 0
    total_prompt_tokens = response.usage.input_tokens + cache_read + cache_creation

    usage = CompletionUsage(
        completion_tokens=response.usage.output_tokens,
        prompt_tokens=total_prompt_tokens,
        total_tokens=total_prompt_tokens + response.usage.output_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cache_read) if cache_read else None,
        cache_creation_input_tokens=cache_creation_value,
    )

    from typing import Literal

    choice = Choice(
        index=0,
        finish_reason=cast(
            "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']", finish_reason or "stop"
        ),
        message=message,
    )

    # The Anthropic Messages API carries no timestamp, so ``created_at`` is absent on a
    # spec-compliant response. ``Message`` does not declare the field either, so an
    # Anthropic-compatible endpoint that sends it anyway has it kept as an unvalidated extra
    # attribute holding the raw JSON value: None for an explicit null, but also int or str.
    # A ``hasattr`` guard passes for all of those and ``.timestamp()`` then raises
    # ``AttributeError: 'NoneType' object has no attribute 'timestamp'``, failing the whole
    # completion, so only fall back to the timestamp when it really is a datetime.
    created_at = getattr(response, "created_at", None)
    created_ts = int(created_at.timestamp()) if isinstance(created_at, datetime) else 0

    return ChatCompletion(
        id=response.id,
        model=response.model,
        created=created_ts,
        object="chat.completion",
        choices=[choice],
        usage=usage,
    )


def _convert_tool_spec(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool specification to Anthropic format."""
    generic_tools = []

    for tool in openai_tools:
        if tool.get("type") != "function":
            continue

        function = tool["function"]
        generic_tool = {
            "name": function["name"],
            "description": function.get("description", ""),
            "parameters": function.get("parameters") or {},
        }
        generic_tools.append(generic_tool)

    anthropic_tools = []
    for tool in generic_tools:
        params: dict[str, Any] = tool["parameters"] or {}
        anthropic_tool = {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": {
                "type": "object",
                "properties": params.get("properties") or {},
                "required": params.get("required", []),
            },
        }
        anthropic_tools.append(anthropic_tool)

    return anthropic_tools


def _convert_tool_choice(params: CompletionParams) -> dict[str, Any]:
    parallel_tool_calls = params.parallel_tool_calls
    if parallel_tool_calls is None:
        parallel_tool_calls = True
    tool_choice = params.tool_choice or "auto"
    if tool_choice == "required":
        tool_choice = "any"
    elif isinstance(tool_choice, dict):
        if tool_choice_type := tool_choice.get("type"):
            if tool_choice_type in ("custom", "function"):
                return {"type": "tool", "name": tool_choice[tool_choice_type]["name"]}
        msg = f"Unsupported tool_choice format: {tool_choice}"
        raise ValueError(msg)
    return {"type": tool_choice, "disable_parallel_tool_use": not parallel_tool_calls}


def _convert_response_format(response_format: dict[str, Any] | type, provider_name: str) -> dict[str, Any]:
    """Convert any-llm response_format to Anthropic's output_config."""
    if is_structured_output_type(response_format):
        schema = get_json_schema(response_format)
    elif isinstance(response_format, dict):
        if response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]["schema"]
        elif response_format.get("type") == "json_object":
            msg = "response_format with type 'json_object'"
            raise UnsupportedParameterError(
                msg,
                provider_name,
                "Use a Pydantic model or json_schema format instead.",
            )
        else:
            msg = f"Unsupported response_format type: {response_format.get('type')}"
            raise ValueError(msg)
    else:
        msg = f"Unsupported response_format: {response_format}"
        raise ValueError(msg)

    return {"format": {"type": "json_schema", "schema": transform_schema(schema)}}


def _convert_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
    """Convert CompletionParams to kwargs for Anthropic API."""
    provider_name: str = kwargs.pop("provider_name")
    result_kwargs: dict[str, Any] = kwargs.copy()

    if params.response_format:
        result_kwargs["output_config"] = _convert_response_format(params.response_format, provider_name)
    if params.max_tokens is None:
        logger.warning(f"max_tokens is required for Anthropic, setting to {DEFAULT_MAX_TOKENS}")
        params.max_tokens = DEFAULT_MAX_TOKENS

    if params.tools:
        params.tools = _convert_tool_spec(params.tools)

    if params.tool_choice is not None or params.parallel_tool_calls is not None:
        params.tool_choice = _convert_tool_choice(params)

    if params.reasoning_effort is None or params.reasoning_effort == "none":
        result_kwargs["thinking"] = {"type": "disabled"}
    elif params.reasoning_effort != "auto":
        result_kwargs["thinking"] = {"type": "adaptive"}
        effort = REASONING_EFFORT_TO_ANTHROPIC_EFFORT[params.reasoning_effort]
        output_config = result_kwargs.get("output_config", {})
        output_config["effort"] = effort
        result_kwargs["output_config"] = output_config

    result_kwargs.update(
        params.model_dump(
            exclude_none=True,
            exclude={
                "model_id",
                "messages",
                "reasoning_effort",
                "response_format",
                "parallel_tool_calls",
                "stream_options",
            },
        )
    )
    result_kwargs["model"] = params.model_id

    if "stop" in result_kwargs:
        stop = result_kwargs.pop("stop")
        result_kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else stop

    system_message, filtered_messages = _convert_messages_for_anthropic(params.messages)
    if system_message:
        result_kwargs["system"] = system_message
    result_kwargs["messages"] = filtered_messages

    return result_kwargs


def _convert_models_list(models_list: list[AnthropicModelInfo]) -> list[Model]:
    """Convert Anthropic models list to OpenAI format.

    ``created_at`` is required by ``ModelInfo``, but an Anthropic-compatible
    proxy or gateway may serve ``/v1/models`` in the OpenAI shape, which carries
    an integer ``created`` and no ``created_at``. The SDK constructs the model
    without validation, so the field is present but ``None``, and calling
    ``.timestamp()`` on it raises ``AttributeError: 'NoneType' object has no
    attribute 'timestamp'`` for the whole listing. Fall back to ``0`` for that
    entry instead, matching what ``_convert_response`` already does for a
    response missing the same field.
    """
    return [
        Model(
            id=model.id,
            object="model",
            created=int(model.created_at.timestamp()) if model.created_at is not None else 0,
            owned_by="anthropic",
        )
        for model in models_list
    ]
