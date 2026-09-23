"""Bidirectional conversion between Anthropic Messages API and OpenAI Chat Completions formats."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from anthropic.types import CacheCreation, CodeExecutionToolResultBlock, ServerToolUseBlock

from any_llm.exceptions import InvalidRequestError
from any_llm.types.completion import ChatCompletionMessageFunctionToolCall
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


def _output_config_to_response_format(output_config: dict[str, Any]) -> dict[str, Any] | None:
    """Translate a raw Anthropic ``output_config`` dict into a completion ``response_format``.

    Lets the bridge carry a non-Pydantic JSON schema to non-Anthropic providers: the schema
    under ``output_config["format"]["schema"]`` is rewrapped as the OpenAI ``json_schema``
    response format. The name falls back to the schema's ``title`` (or ``"structured_output"``).

    Both dict shapes ``normalize_output_config`` accepts are handled. ``None`` means the config
    asked for no structured output, so the caller leaves ``response_format`` unset.

    ``output_config.effort`` is never translated: chat completions has no equivalent, and the
    nearest field, ``reasoning_effort``, governs reasoning rather than output. It is ignored
    whether or not a schema sits beside it, so an effort-only config is not rejected in one
    shape while being dropped in the other.

    Raises:
        InvalidRequestError: when the config names a format but no usable schema. The caller
            asked for structured output, and a ``response_format`` carrying an empty schema
            would sit on the wire constraining nothing while they believe it is in force.

    """
    fmt = normalize_output_config(output_config).get("format")
    if fmt is None:
        return None
    schema = fmt.get("schema") if isinstance(fmt, dict) else None
    if not isinstance(schema, dict) or not schema:
        msg = (
            "output_format names a format but carries no JSON schema. Expected an Anthropic "
            'output_config ({"format": {"type": "json_schema", "schema": {...}}}) or the bare '
            'format object ({"type": "json_schema", "schema": {...}}).'
        )
        raise InvalidRequestError(msg)
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
        elif (
            response_format := _output_config_to_response_format(cast("dict[str, Any]", params.output_format))
        ) is not None:
            result["response_format"] = response_format

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

    The Anthropic ``signature`` travels in the ``extra_content["anthropic"]`` side-channel that
    ``anthropic``'s ``_extract_anthropic_thinking_signature`` already reads, so a bridged
    request that later reaches an Anthropic-native provider can rebuild the block whole.
    Anthropic requires that signature back unmodified while extended thinking is on.

    Neither key reaches every backend as emitted here. ``strip_extra_content`` drops the
    ``anthropic`` side-channel in ``BaseOpenAIProvider`` (but not ``otari``, which overrides
    ``_acompletion``), ``groq`` and ``cerebras``, and ``groq`` and ``cerebras``, whose APIs name
    the field ``reasoning``, rename ``reasoning_content`` through
    ``replay_reasoning_content_as_reasoning``.

    A signature is emitted only when the turn holds a single ``thinking`` block. Interleaved
    thinking can put several in one turn, and the OpenAI wire has one ``reasoning_content``
    string to hold them, so the joined text is not what any one signature signs. Emitting one
    anyway would pair a signature with text it does not cover, which Anthropic rejects on
    replay; the text is kept either way, since that is what the backend reads.

    ``redacted_thinking`` blocks are dropped. They carry encrypted payloads with no text to
    join and nothing on the OpenAI wire to carry them, so preserving them needs a side-channel
    schema of its own.

    Gemini state rides the ``extra_content["google"]`` side-channel that ``gemini`` reads, the
    inverse of what ``chat_completion_to_message_response`` puts on the blocks. A ``tool_use``
    block's ``extra_content`` (the function call's ``thought_signature``) becomes its tool call's
    ``extra_content`` verbatim. A ``text`` block's ``extra_content["google"]`` merges into the
    message's, beside the ``anthropic`` signature rather than in place of it. ``server_tool_use``
    blocks named ``code_execution`` and ``code_execution_tool_result`` blocks become
    ``extra_content["google"]["code_execution"]`` items in block order, without the files the code
    produced (``inline_outputs``, or ``file_id`` blocks in the result's ``content``): those are
    output, not something to replay. Any other server tool block has no chat completions
    representation and is dropped.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    signature: str | None = None
    tool_calls: list[dict[str, Any]] = []
    google: dict[str, Any] = {}
    code_execution: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
            block_extra = block.get("extra_content")
            if isinstance(block_extra, dict) and isinstance(block_extra.get("google"), dict):
                google.update({k: v for k, v in block_extra["google"].items() if k != "code_execution"})
        elif block_type == "thinking":
            thinking_parts.append(block.get("thinking", ""))
            block_signature = block.get("signature")
            if isinstance(block_signature, str) and block_signature:
                signature = block_signature
        elif block_type == "tool_use":
            tool_call: dict[str, Any] = {
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            }
            if isinstance(block.get("extra_content"), dict):
                tool_call["extra_content"] = block["extra_content"]
            tool_calls.append(tool_call)
        elif block_type == "server_tool_use" and block.get("name") == "code_execution":
            code_execution.append(_server_tool_use_to_executable_code(block))
        elif block_type == "code_execution_tool_result":
            code_execution.append(_code_execution_tool_result_to_item(block))

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
    extra_content: dict[str, Any] = {}
    if signature is not None and len(thinking_parts) == 1:
        extra_content["anthropic"] = {"signature": signature}
    if code_execution:
        google["code_execution"] = code_execution
    if google:
        extra_content["google"] = google
    if extra_content:
        result["extra_content"] = extra_content
    return [result]


def _server_tool_use_to_executable_code(block: dict[str, Any]) -> dict[str, Any]:
    """Convert a ``code_execution`` ``server_tool_use`` block to a Gemini ``executable_code`` item."""
    tool_input = block.get("input")
    if not isinstance(tool_input, dict):
        tool_input = {}
    return {
        "type": "executable_code",
        "id": block.get("id", ""),
        "language": tool_input.get("language") or "PYTHON",
        "code": tool_input.get("code", ""),
    }


def _code_execution_tool_result_to_item(block: dict[str, Any]) -> dict[str, Any]:
    """Convert a ``code_execution_tool_result`` block to a Gemini ``code_execution_result`` item.

    Gemini reports one ``output`` string and an outcome, so stdout and stderr are joined and a
    nonzero return code, or an error result, reads as ``OUTCOME_FAILED``. An error result has no
    output, so its ``error_code`` stands in for one.
    """
    content = block.get("content")
    if not isinstance(content, dict):
        content = {}
    if content.get("type") == "code_execution_tool_result_error":
        outcome = "OUTCOME_FAILED"
        output = str(content.get("error_code", ""))
    else:
        outcome = "OUTCOME_OK" if content.get("return_code") == 0 else "OUTCOME_FAILED"
        output = f"{content.get('stdout', '')}{content.get('stderr', '')}"
    return {
        "type": "code_execution_result",
        "id": block.get("tool_use_id", ""),
        "outcome": outcome,
        "output": output,
    }


def _convert_image_block_to_openai(block: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic ``image`` block to an OpenAI ``image_url`` content part.

    Inverse of the ``image_url`` branch in ``anthropic``'s ``_convert_content_for_anthropic``.

    Raises:
        InvalidRequestError: when the source carries neither inline data nor a url, matching
            what ``_convert_document_block_to_openai`` does. An empty ``image_url.url`` is not
            an attachment a backend can fetch.

    """
    source = block.get("source", {})
    if source.get("type") == "base64":
        data = source.get("data", "")
        if not data:
            msg = "image block base64 source carries no data"
            raise InvalidRequestError(msg)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{source.get('media_type', 'image/png')};base64,{data}"},
        }
    url = source.get("url", "")
    if not url:
        msg = f"image block source carries no payload (source type {source.get('type')!r})"
        raise InvalidRequestError(msg)
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
        data = source.get("data", "")
        if not data:
            msg = "document block base64 source carries no data"
            raise InvalidRequestError(msg)
        media_type = source.get("media_type", "application/pdf")
        return {"type": "file", "file": {"file_data": f"data:{media_type};base64,{data}"}}
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
    after_rendered_block = False
    for block in tool_content:
        block_type = block.get("type", "")
        if block_type == "text":
            text = block.get("text", "")
            if after_rendered_block and text:
                text_parts.append("\n")
                after_rendered_block = False
            text_parts.append(text)
        elif block_type == "image":
            extra_parts.append(_convert_image_block_to_openai(block))
        elif block_type == "document":
            extra_parts.append(_convert_document_block_to_openai(block))
        elif rendered := _render_tool_result_block_as_text(block):
            if any(text_parts):
                text_parts.append("\n")
            text_parts.append(rendered)
            after_rendered_block = True
    return "".join(text_parts), extra_parts


def _render_tool_result_block_as_text(block: dict[str, Any]) -> str | None:
    """Render a tool result block that has no OpenAI part as text, or return ``None`` for an unknown type.

    Anthropic also allows ``search_result``, ``tool_reference`` and ``browser_state`` blocks in a
    ``tool_result``. None has an OpenAI equivalent, and dropping them loses the tool's output, so
    each becomes text on the ``role: tool`` message, set off from neighbouring text by newlines.
    Anthropic renders ``browser_state`` into model-visible text server-side; no other backend
    does, so its fields are sent as JSON.
    """
    block_type = block.get("type", "")
    if block_type == "search_result":
        body = "".join(part.get("text", "") for part in block.get("content", []) if part.get("type") == "text")
        return "\n".join(part for part in (block.get("title", ""), block.get("source", ""), body) if part)
    if block_type == "tool_reference":
        return f"Tool reference: {block.get('tool_name', '')}"
    if block_type == "browser_state":
        return json.dumps({key: block[key] for key in ("tabs", "state_changes") if key in block})
    return None


def _convert_user_blocks_to_openai(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic user content blocks to OpenAI format.

    Handles tool_result blocks (→ role:tool messages) and content blocks (text, image).

    A tool result marked ``is_error`` has its text prefixed with ``Error: ``, unless it already
    starts with one, rather than carrying the flag as a message key. OpenAI has no field for it, the OpenAI SDK forwards unknown message
    keys verbatim, and strict OpenAI-compatible backends such as Fireworks reject the whole request
    over an unknown key. The text is the one place the signal reaches the model on every backend,
    including providers that rebuild the message from known keys, as ``bedrock``, ``gemini`` and
    ``ollama`` do. Mapping it onto a native representation, such as ``toolResult.status`` on
    Bedrock, is left to a follow-up.

    A tool result carrying image or document blocks emits the text as the ``role: tool``
    message and holds the remaining parts back, because OpenAI accepts text only on a tool
    message. The held parts lead the ``user`` message that closes the turn, so they stay at the
    same point in the conversation rather than being dropped.

    They are held until the whole run of tool results ends rather than emitted after each one.
    Anthropic puts every ``tool_result`` of a parallel tool call in a single user turn, so
    emitting per result would interleave user messages between the ``role: tool`` messages, and
    OpenAI requires those to follow the assistant ``tool_calls`` turn with nothing in between.
    """
    results: list[dict[str, Any]] = []
    content_blocks: list[dict[str, Any]] = []
    held_parts: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type", "")
        if block_type == "tool_result":
            # Flush any accumulated content blocks first
            if content_blocks:
                results.append({"role": "user", "content": content_blocks})
                content_blocks = []
            tool_text, extra_parts = _convert_tool_result_content(block.get("content", ""))
            if block.get("is_error") is True:
                if not tool_text:
                    tool_text = "Error"
                elif not tool_text.startswith("Error:"):
                    tool_text = f"Error: {tool_text}"
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": tool_text,
                }
            )
            held_parts.extend(extra_parts)
        elif block_type == "text":
            content_blocks.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "image":
            content_blocks.append(_convert_image_block_to_openai(block))
        else:
            content_blocks.append(block)

    if held_parts or content_blocks:
        results.append({"role": "user", "content": held_parts + content_blocks})

    return results


def _convert_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool format to OpenAI function tool format.

    A tool with none of ``name``, ``type`` and ``input_schema`` is not an Anthropic tool but a
    provider-native one, such as Gemini's ``{"google_search": {}}`` or ``{"code_execution": {}}``,
    and is forwarded as-is for the provider to read. Converting it would send a function with no
    name and an empty schema.
    """
    openai_tools = []
    for tool in tools:
        if not {"name", "type", "input_schema"} & tool.keys():
            openai_tools.append(tool)
            continue
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


def split_cached_input_tokens(
    prompt_tokens: int,
    cached_tokens: int | None,
    cache_write_tokens: int | None = None,
) -> tuple[int, int | None]:
    """Split an OpenAI prompt-token total into disjoint Anthropic input/cache-read counts.

    OpenAI reports ``prompt_tokens`` as the whole prompt, with ``prompt_tokens_details.cached_tokens`` and
    ``prompt_tokens_details.cache_write_tokens`` as subsets of it, while Anthropic's ``input_tokens``,
    ``cache_read_input_tokens`` and ``cache_creation_input_tokens`` are disjoint and sum to the prompt.
    Copying the cache counts across without subtracting would make any consumer that sums the fields
    over-count, and would bill cached tokens twice in a cost model that prices them at different rates.

    The cached count comes back as ``None`` when the provider reported no read meter, and as 0 when it
    reported an explicit zero, so consumers can tell "no cache hit" from "no cache accounting".

    Each cache count is clamped into what remains of ``prompt_tokens`` so a provider that reports them
    inconsistently cannot push ``input_tokens`` negative or above the prompt total. Clamping the
    subtrahends rather than flooring the result keeps the returned input and cached counts summing to
    ``prompt_tokens`` minus the cache writes.
    """
    remaining = prompt_tokens - min(max(cache_write_tokens or 0, 0), prompt_tokens)
    cached = min(max(cached_tokens or 0, 0), remaining)
    cache_read = cached if cached_tokens is not None and (cached_tokens == 0 or cached > 0) else None
    return remaining - cached, cache_read


def _cached_tokens_from_usage(usage: CompletionUsage) -> int | None:
    """Read ``prompt_tokens_details.cached_tokens``, preserving absent versus zero."""
    details = usage.prompt_tokens_details
    return details.cached_tokens if details is not None else None


def _cache_write_tokens_from_usage(usage: CompletionUsage) -> int | None:
    details = usage.prompt_tokens_details
    return details.cache_write_tokens if details is not None else None


def _cache_creation_details_from_usage(usage: CompletionUsage) -> CacheCreation | None:
    """Build Anthropic's TTL breakdown from ``prompt_tokens_details.cache_creation_token_details``.

    ``CacheCreation`` requires both buckets, so a usage that reports only one yields ``None`` rather than a
    fabricated zero; the write total still travels on ``cache_creation_input_tokens``.
    """
    details = usage.prompt_tokens_details
    ttl = details.cache_creation_token_details if details is not None else None
    if ttl is None or ttl.ephemeral_5m_input_tokens is None or ttl.ephemeral_1h_input_tokens is None:
        return None
    return CacheCreation(
        ephemeral_5m_input_tokens=ttl.ephemeral_5m_input_tokens,
        ephemeral_1h_input_tokens=ttl.ephemeral_1h_input_tokens,
    )


def _google_extra_content(extra_content: dict[str, Any] | None) -> dict[str, Any]:
    """Return the ``google`` namespace of an ``extra_content`` side-channel, or an empty dict."""
    google = (extra_content or {}).get("google")
    return google if isinstance(google, dict) else {}


def _code_execution_items(extra_content: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the Gemini ``code_execution`` items of an ``extra_content`` side-channel, skipping malformed ones."""
    items = _google_extra_content(extra_content).get("code_execution")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _executable_code_block(item: dict[str, Any]) -> ServerToolUseBlock:
    """Map an ``executable_code`` item to a ``server_tool_use`` block named ``code_execution``."""
    return ServerToolUseBlock(
        type="server_tool_use",
        id=item.get("id") or "",
        name="code_execution",
        input={"code": item.get("code") or "", "language": item.get("language") or "PYTHON"},
    )


def _code_execution_result_block(item: dict[str, Any], outputs: list[dict[str, Any]]) -> CodeExecutionToolResultBlock:
    """Map a ``code_execution_result`` item, and the files its code produced, to a ``code_execution_tool_result`` block.

    Gemini reports one ``output`` string, so it lands in stdout when the outcome is ``OUTCOME_OK``
    and in stderr otherwise, with the return code standing in for the outcome. Produced files
    ride an ``inline_outputs`` extra field as ``{"mime_type", "data"}`` entries: Anthropic names
    them by ``file_id`` in ``content``, which only a file store can mint, so a layer that has one
    swaps them in.
    """
    ok = item.get("outcome") == "OUTCOME_OK"
    output = item.get("output") or ""
    result: dict[str, Any] = {
        "type": "code_execution_result",
        "stdout": output if ok else "",
        "stderr": "" if ok else output,
        "return_code": 0 if ok else 1,
        "content": [],
    }
    if outputs:
        result["inline_outputs"] = outputs
    return CodeExecutionToolResultBlock.model_validate(
        {"type": "code_execution_tool_result", "tool_use_id": item.get("id") or "", "content": result}
    )


def _inline_output(item: dict[str, Any]) -> dict[str, Any]:
    return {"mime_type": item.get("mime_type") or "", "data": item.get("data") or ""}


def _code_execution_blocks(extra_content: dict[str, Any] | None) -> list[MessageContentBlock]:
    """Map Gemini ``code_execution`` items to Anthropic code execution blocks.

    An ``executable_code`` item becomes a ``server_tool_use`` block and a ``code_execution_result``
    item a ``code_execution_tool_result`` block. A ``code_execution_output`` item (a file the code
    produced) folds into the result with its id, or the last result when none matches, and is
    dropped when there is no result at all.
    """
    items = _code_execution_items(extra_content)
    outputs_by_id: dict[str, list[dict[str, Any]]] = {}
    last_outputs: list[dict[str, Any]] | None = None
    for item in items:
        if item.get("type") == "code_execution_result":
            last_outputs = outputs_by_id[item.get("id") or ""] = []
    for item in items:
        if item.get("type") == "code_execution_output":
            target = outputs_by_id.get(item.get("id") or "", last_outputs)
            if target is not None:
                target.append(_inline_output(item))

    blocks: list[MessageContentBlock] = []
    for item in items:
        if item.get("type") == "executable_code":
            blocks.append(_executable_code_block(item))
        elif item.get("type") == "code_execution_result":
            blocks.append(_code_execution_result_block(item, outputs_by_id[item.get("id") or ""]))
    return blocks


def _text_block(text: str, thought_signature: Any) -> TextBlock:
    """Build a text block, carrying a Gemini text-part ``thought_signature`` in ``extra_content``."""
    if isinstance(thought_signature, str) and thought_signature:
        return TextBlock.model_validate(
            {"type": "text", "text": text, "extra_content": {"google": {"thought_signature": thought_signature}}}
        )
    return TextBlock(type="text", text=text)


def chat_completion_to_message_response(completion: ChatCompletion) -> MessageResponse:
    """Convert an OpenAI ChatCompletion to an Anthropic MessageResponse.

    Blocks come out in the order thinking, code execution, text, refusal, tool use. Gemini state
    in the ``extra_content`` side-channel is kept on the blocks as an ``extra_content`` field,
    which ``_convert_assistant_blocks_to_openai`` reads back when the turn is replayed: a tool
    call's on its ``tool_use`` block, and the message-level ``thought_signature`` (Gemini signs
    the last text part) on the text block. No text block is added to carry that signature alone,
    except the empty one a response with no other content gets anyway. Code execution items
    become ``server_tool_use`` and ``code_execution_tool_result`` blocks (see
    ``_code_execution_blocks``).
    """
    content_blocks: list[MessageContentBlock] = []
    stop_reason: StopReason = "end_turn"
    text_signature: Any = None

    if completion.choices:
        choice = completion.choices[0]
        msg = choice.message
        text_signature = _google_extra_content(msg.extra_content).get("thought_signature")

        if msg.reasoning:
            content_blocks.append(ThinkingBlock(type="thinking", thinking=msg.reasoning.content))

        content_blocks.extend(_code_execution_blocks(msg.extra_content))

        if msg.content:
            content_blocks.append(_text_block(msg.content, text_signature))
            text_signature = None

        if msg.refusal:
            content_blocks.append(TextBlock(type="text", text=msg.refusal))

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
                    _tool_use_block(
                        tc.id,
                        fn.name,
                        tool_input,
                        tc.extra_content if isinstance(tc, ChatCompletionMessageFunctionToolCall) else None,
                    )
                )

        stop_reason = "refusal" if msg.refusal else _finish_reason_to_stop_reason(choice.finish_reason)

    if not content_blocks:
        content_blocks.append(_text_block("", text_signature))

    usage = MessageUsage(input_tokens=0, output_tokens=0)
    if completion.usage:
        input_tokens, cache_read = split_cached_input_tokens(
            completion.usage.prompt_tokens,
            _cached_tokens_from_usage(completion.usage),
            _cache_write_tokens_from_usage(completion.usage),
        )
        usage = MessageUsage(
            input_tokens=input_tokens,
            cache_read_input_tokens=cache_read,
            output_tokens=completion.usage.completion_tokens,
            cache_creation_input_tokens=_cache_write_tokens_from_usage(completion.usage),
            cache_creation=_cache_creation_details_from_usage(completion.usage),
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


def _tool_use_block(
    tool_id: str, name: str, tool_input: dict[str, Any], extra_content: dict[str, Any] | None
) -> ToolUseBlock:
    """Build a tool_use block, carrying the tool call's ``extra_content`` when it has one."""
    if extra_content:
        return ToolUseBlock.model_validate(
            {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input, "extra_content": extra_content}
        )
    return ToolUseBlock(type="tool_use", id=tool_id, name=name, input=tool_input)


def _finish_reason_to_stop_reason(finish_reason: str | None) -> StopReason:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    mapping: dict[str, StopReason] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
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
        self.cache_read_input_tokens: int | None = None
        self.cache_creation_input_tokens: int | None = None
        self.cache_creation: CacheCreation | None = None
        self.stop_reason: StopReason | None = None
        self.tool_call_id: str | None = None
        self.tool_call_name: str | None = None
        self.tool_block_indexes: dict[int, int] = {}
        """Content block index of each open tool_use block, keyed by OpenAI ``tool_calls[].index``."""
        self.pending_code_result: tuple[dict[str, Any], list[dict[str, Any]]] | None = None
        """Held-back ``code_execution_result`` item and the inline outputs folded into it so far."""
        self.code_result_ids: set[str] = set()


def chat_completion_chunk_to_message_stream_events(
    chunk: ChatCompletionChunk,
    state: StreamingState,
) -> list[MessageStartEvent | ContentBlockStartEvent | ContentBlockDeltaEvent | ContentBlockStopEvent]:
    """Convert a ChatCompletionChunk to a list of MessageStreamEvents.

    This is stateful: it tracks the current content block index and type to emit
    the correct lifecycle events (start/delta/stop).

    Gemini state follows the non-streaming conversion where the event stream has room for it. A
    tool_use ``content_block_start`` carries the tool call's ``extra_content`` when the fragment
    that opens the block has one. Code execution items each become a complete block, opened with
    its full input or result on ``content_block_start`` and closed at once, with no
    ``input_json_delta``: consumers that rebuild blocks from deltas typically do so for
    ``tool_use`` only, and would otherwise re-serialize a ``server_tool_use`` with empty input.
    A ``code_execution_tool_result`` is held back until something other than a
    ``code_execution_output`` arrives (or the stream ends), because the files its code produced
    can come in later chunks and have to be on its start event.

    A message-level ``thought_signature`` is dropped. Gemini signs the last text part, and by the
    time the signature arrives that text block has already started, so no Anthropic event can
    attach it. Google makes text-part signatures optional; the mandatory function-call
    signatures travel on the tool_use blocks.
    """
    events: list[MessageStartEvent | ContentBlockStartEvent | ContentBlockDeltaEvent | ContentBlockStopEvent] = []
    state.model = chunk.model

    if chunk.usage:
        if chunk.usage.prompt_tokens:
            state.input_tokens = chunk.usage.prompt_tokens
        if chunk.usage.completion_tokens:
            state.output_tokens = chunk.usage.completion_tokens
        cached = _cached_tokens_from_usage(chunk.usage)
        if cached is not None:
            state.cache_read_input_tokens = cached
        cache_write = _cache_write_tokens_from_usage(chunk.usage)
        if cache_write is not None:
            state.cache_creation_input_tokens = cache_write
        cache_creation_details = _cache_creation_details_from_usage(chunk.usage)
        if cache_creation_details is not None:
            state.cache_creation = cache_creation_details

    if not state.started:
        state.started = True
        input_tokens, cache_read = split_cached_input_tokens(
            state.input_tokens,
            state.cache_read_input_tokens,
            state.cache_creation_input_tokens,
        )
        usage = MessageUsage(
            input_tokens=input_tokens,
            cache_read_input_tokens=cache_read,
            output_tokens=0,
            cache_creation_input_tokens=state.cache_creation_input_tokens,
            cache_creation=state.cache_creation,
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

    for item in _code_execution_items(delta.extra_content):
        item_type = item.get("type")
        item_id = item.get("id") or ""
        if item_type == "code_execution_output":
            pending = state.pending_code_result
            # Like the non-streaming fold: the result with this id, else the last one, which is
            # the held one unless the matching result has already been emitted.
            if pending is not None and (item_id == pending[0].get("id") or item_id not in state.code_result_ids):
                pending[1].append(_inline_output(item))
        elif item_type == "executable_code":
            _close_current_block(state, events)
            state.current_block_index += 1
            events.append(
                ContentBlockStartEvent(
                    type="content_block_start",
                    index=state.current_block_index,
                    content_block=_executable_code_block(item),
                )
            )
            events.append(ContentBlockStopEvent(type="content_block_stop", index=state.current_block_index))
        elif item_type == "code_execution_result":
            _close_current_block(state, events)
            state.pending_code_result = (item, [])
            state.code_result_ids.add(item_id)

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

    if delta.refusal:
        state.stop_reason = "refusal"
        # A distinct block type keeps refusal text out of the block holding any partial answer,
        # matching the separate TextBlock the non-streaming conversion produces.
        if state.current_block_type != "refusal":
            _close_current_block(state, events)
            state.current_block_index += 1
            state.current_block_type = "refusal"
            events.append(
                ContentBlockStartEvent(
                    type="content_block_start",
                    index=state.current_block_index,
                    content_block=TextBlock(type="text", text=""),
                )
            )
        events.append(
            ContentBlockDeltaEvent(
                type="content_block_delta",
                index=state.current_block_index,
                delta=TextDelta(type="text_delta", text=delta.refusal),
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
                        content_block=_tool_use_block(
                            state.tool_call_id or "", state.tool_call_name or "", {}, tc.extra_content
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
        # OpenAI ends a streamed refusal with finish_reason="stop", which must not mask the refusal.
        if state.stop_reason != "refusal":
            state.stop_reason = _finish_reason_to_stop_reason(choice.finish_reason)

    return events


def close_open_blocks(state: StreamingState) -> list[ContentBlockStartEvent | ContentBlockStopEvent]:
    """Build a content_block_stop event for every block still open, in block order.

    A tool_use section can hold more than one open block, because each parallel tool call gets
    its own block and stays open until the section ends. A held-back code execution result is
    emitted after them, as a start and a stop, since nothing can be added to it any more.
    """
    events: list[ContentBlockStartEvent | ContentBlockStopEvent] = []
    if state.current_block_type is not None:
        open_indexes = sorted(state.tool_block_indexes.values()) or [state.current_block_index]
        state.tool_block_indexes.clear()
        state.current_block_type = None
        events.extend(ContentBlockStopEvent(type="content_block_stop", index=index) for index in open_indexes)
    if state.pending_code_result is not None:
        item, outputs = state.pending_code_result
        state.pending_code_result = None
        state.current_block_index += 1
        events.append(
            ContentBlockStartEvent(
                type="content_block_start",
                index=state.current_block_index,
                content_block=_code_execution_result_block(item, outputs),
            )
        )
        events.append(ContentBlockStopEvent(type="content_block_stop", index=state.current_block_index))
    return events


def _close_current_block(
    state: StreamingState,
    events: list[MessageStartEvent | ContentBlockStartEvent | ContentBlockDeltaEvent | ContentBlockStopEvent],
) -> None:
    """Emit content_block_stop events for any open blocks, and the held-back code execution result."""
    events.extend(close_open_blocks(state))


_ANTHROPIC_SERVER_TOOL_ID_PREFIX = "srvtoolu_"


def prepare_blocks_for_native_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ``messages`` with the bridge's content blocks made acceptable to a native Anthropic Messages API.

    Two things the Messages bridge puts in a conversation would be rejected there. The
    ``extra_content`` field (any_llm's side-channel for provider state, see
    ``chat_completion_to_message_response``) is an unknown block field, so it is removed. Code
    execution blocks mapped from another provider's items carry ids Anthropic did not issue
    (its own start with ``srvtoolu_``), so each ``server_tool_use`` named ``code_execution`` and
    its ``code_execution_tool_result`` become one text block holding the code and its output,
    which keeps the context for the model; a use or result without its other half becomes the
    same text without the missing part. Anthropic's own code execution blocks pass untouched.

    Messages and blocks that need no change are returned as the same objects; the others are
    copied, so the caller's dicts are never mutated.
    """
    prepared: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            blocks = _prepare_blocks_for_native_messages(content)
            if blocks is not None:
                message = {**message, "content": blocks}
        prepared.append(message)
    return prepared


def _is_foreign_code_execution(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    if block.get("type") == "server_tool_use" and block.get("name") == "code_execution":
        block_id = block.get("id")
    elif block.get("type") == "code_execution_tool_result":
        block_id = block.get("tool_use_id")
    else:
        return False
    return not (isinstance(block_id, str) and block_id.startswith(_ANTHROPIC_SERVER_TOOL_ID_PREFIX))


def _prepare_blocks_for_native_messages(blocks: list[Any]) -> list[Any] | None:
    """Return the prepared blocks of one message, or ``None`` when none needs a change."""
    if not any(isinstance(b, dict) and ("extra_content" in b or _is_foreign_code_execution(b)) for b in blocks):
        return None
    results = {
        b.get("tool_use_id"): b
        for b in blocks
        if _is_foreign_code_execution(b) and b.get("type") == "code_execution_tool_result"
    }
    use_ids = {b.get("id") for b in blocks if _is_foreign_code_execution(b) and b.get("type") == "server_tool_use"}
    prepared: list[Any] = []
    for block in blocks:
        if not isinstance(block, dict):
            prepared.append(block)
        elif _is_foreign_code_execution(block) and block.get("type") == "server_tool_use":
            prepared.append({"type": "text", "text": _code_execution_as_text(block, results.get(block.get("id")))})
        elif _is_foreign_code_execution(block):
            if block.get("tool_use_id") not in use_ids:
                prepared.append({"type": "text", "text": _code_execution_as_text(None, block)})
        else:
            prepared.append({k: v for k, v in block.items() if k != "extra_content"})
    return prepared


def _code_execution_as_text(use: dict[str, Any] | None, result: dict[str, Any] | None) -> str:
    """Render a code execution use and its result, either of which may be missing, as text."""
    parts: list[str] = []
    if use is not None:
        tool_input = use.get("input")
        if not isinstance(tool_input, dict):
            tool_input = {}
        language = str(tool_input.get("language") or "python").lower()
        parts.append(f"Code execution ({language}):\n```{language}\n{tool_input.get('code', '')}\n```")
    if result is not None:
        content = result.get("content")
        if not isinstance(content, dict):
            content = {}
        if content.get("type") == "code_execution_tool_result_error":
            output = f"Error: {content.get('error_code', '')}"
        else:
            output = f"{content.get('stdout', '')}{content.get('stderr', '')}"
        parts.append(f"Output:\n{output}")
    return "\n".join(parts)
