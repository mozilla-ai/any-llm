import base64
import binascii
import json
import mimetypes
from contextlib import suppress
from time import time
from typing import Any, Literal, cast

from google.genai import types
from google.genai.pagers import Pager
from pydantic import ValidationError

from any_llm.exceptions import InvalidRequestError, UnsupportedParameterError
from any_llm.logging import logger
from any_llm.types.batch import Batch, BatchRequestCounts, BatchResult, BatchResultError, BatchResultItem
from any_llm.types.completion import (
    AudioContent,
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    CompletionTokensDetails,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    ImageContent,
    PromptTokensDetails,
    Reasoning,
    Usage,
)
from any_llm.types.model import Model

_INLINE_SIZE_LIMIT = 20 * 1024 * 1024
_GEMINI_CONTENT_FILTER_REFUSAL = "Response blocked by Gemini content filtering."


def _has_json_schema_refs(schema: Any) -> bool:
    """Return True if *schema* contains a ``$defs`` block or any ``$ref`` reference.

    ``google.genai.types.Schema`` does not accept ``$ref``/``$defs``. When present,
    the schema must be routed through ``FunctionDeclaration.parameters_json_schema``
    instead, which the SDK forwards to the server as raw JSON Schema.
    """
    if isinstance(schema, dict):
        if "$defs" in schema or "$ref" in schema:
            return True
        return any(_has_json_schema_refs(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_json_schema_refs(v) for v in schema)
    return False


def _has_type_unions(schema: Any) -> bool:
    """Return True if *schema* declares a type as a list, e.g. ``["string", "null"]``.

    ``google.genai.types.Schema.type`` is a single ``types.Type`` enum, so the OpenAPI
    3.0 shape cannot express a union and the API rejects the request. JSON Schema does
    allow the list form, so such schemas must be routed through
    ``FunctionDeclaration.parameters_json_schema``, which the SDK forwards to the
    server as raw JSON Schema.
    """
    if isinstance(schema, dict):
        if isinstance(schema.get("type"), list):
            return True
        return any(_has_type_unions(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_type_unions(v) for v in schema)
    return False


def _has_additional_properties(schema: Any) -> bool:
    """Return True if *schema* sets ``additionalProperties`` anywhere, nested included.

    ``google.genai.types.Schema`` serializes the key as ``additional_properties``, which the
    Gemini API rejects with a 400. Such schemas must be routed through
    ``GenerateContentConfig.response_json_schema``, which the SDK forwards as raw JSON Schema.
    """
    if isinstance(schema, dict):
        if "additionalProperties" in schema:
            return True
        return any(_has_additional_properties(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_additional_properties(v) for v in schema)
    return False


def _convert_tool_spec(tools: list[dict[str, Any] | Any], provider_name: str) -> list[types.Tool]:
    converted_tools = []
    function_declarations = []

    for tool in tools:
        if isinstance(tool, types.Tool):
            converted_tools.append(tool)
            continue

        if tool.get("type") != "function":
            # not openai function format: validate as a gemini-native tool dict, e.g. {"google_search": {}}
            try:
                converted_tools.append(types.Tool.model_validate(tool))
            except ValidationError as exc:
                msg = f"Unsupported tool: {tool}"
                raise InvalidRequestError(msg, provider_name=provider_name) from exc
            continue

        function = tool["function"]
        params: dict[str, Any] = function.get("parameters") or {}

        if _has_json_schema_refs(params) or _has_type_unions(params):
            function_declarations.append(
                types.FunctionDeclaration(
                    name=function["name"],
                    description=function.get("description", ""),
                    parameters_json_schema=params,
                )
            )
            continue

        properties: dict[str, dict[str, Any]] = {}
        for param_name, param_info in (params.get("properties") or {}).items():
            prop: dict[str, Any] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            if "enum" in param_info:
                prop["enum"] = param_info["enum"]
            if "items" in param_info:
                prop["items"] = param_info["items"]
            if prop.get("type") == "array" and "items" not in prop:
                prop["items"] = {"type": "string"}
            properties[param_name] = prop

        parameters_dict = {
            "type": "object",
            "properties": properties,
            "required": params.get("required", []),
        }

        function_declarations.append(
            types.FunctionDeclaration(
                name=function["name"],
                description=function.get("description", ""),
                parameters=types.Schema(**parameters_dict),
            )
        )
    if function_declarations:
        converted_tools.append(types.Tool(function_declarations=function_declarations))
    return converted_tools


def _convert_tool_choice(tool_choice: str | dict[str, Any], provider_name: str) -> types.ToolConfig:
    error_message = "tool_choice"
    additional_message = f"Unsupported tool_choice: {tool_choice}"

    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "allowed_tools":
            allowed = tool_choice.get("allowed_tools")
            # Gemini only honors allowed_function_names in ANY mode, so mode="auto" has no equivalent.
            if not isinstance(allowed, dict) or allowed.get("mode") != "required":
                raise UnsupportedParameterError(error_message, provider_name, additional_message)
            allowed_tools = allowed.get("tools")
            if not isinstance(allowed_tools, list):
                raise UnsupportedParameterError(error_message, provider_name, additional_message)
            # Every entry is kept so that an unusable one fails the name check below rather than
            # being dropped, which would silently narrow the set of tools the caller asked for.
            functions = [
                tool.get("function") if isinstance(tool, dict) and tool.get("type") == "function" else None
                for tool in allowed_tools
            ]
        else:
            functions = [tool_choice.get("function")] if tool_choice.get("type") == "function" else []
        raw_names = [function.get("name") if isinstance(function, dict) else None for function in functions]
        names = [name for name in raw_names if isinstance(name, str) and name]
        if not names or len(names) != len(raw_names):
            raise UnsupportedParameterError(error_message, provider_name, additional_message)
        return types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY,
                allowed_function_names=names,
            )
        )

    tool_choice_to_mode = {
        "required": types.FunctionCallingConfigMode.ANY,
        "auto": types.FunctionCallingConfigMode.AUTO,
        "none": types.FunctionCallingConfigMode.NONE,
    }
    mode = tool_choice_to_mode.get(tool_choice)
    if mode is None:
        raise UnsupportedParameterError(error_message, provider_name, additional_message)

    return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode))


def _parse_data_uri(data_uri: str, field_name: str, provider_name: str) -> tuple[str, bytes]:
    if not data_uri.startswith("data:"):
        msg = f"{field_name} must be a data URI"
        raise InvalidRequestError(msg, provider_name=provider_name)
    if "base64," not in data_uri:
        msg = f"{field_name} must be a base64-encoded data URI"
        raise InvalidRequestError(msg, provider_name=provider_name)

    mime_part = data_uri[5:]
    semi_idx = mime_part.find(";")
    mime_type = mime_part[:semi_idx] if semi_idx != -1 else mime_part
    if not mime_type:
        msg = f"{field_name} is missing a MIME type"
        raise InvalidRequestError(msg, provider_name=provider_name)

    encoded_data = data_uri.split("base64,", 1)[1]
    if not encoded_data:
        msg = f"{field_name} is missing base64 data"
        raise InvalidRequestError(msg, provider_name=provider_name)

    try:
        raw_data = base64.b64decode(encoded_data, validate=True)
    except binascii.Error as exc:
        msg = f"{field_name} contains invalid base64 data"
        raise InvalidRequestError(msg, exc, provider_name) from exc
    return mime_type, raw_data


def _validate_inline_size(raw_data: bytes, field_name: str, provider_name: str) -> None:
    if len(raw_data) > _INLINE_SIZE_LIMIT:
        msg = f"{field_name} exceeds the 20 MB inline upload limit for {provider_name} ({len(raw_data)} bytes)"
        raise InvalidRequestError(msg, provider_name=provider_name)


def _convert_image_url_to_part(block: dict[str, Any], provider_name: str) -> types.Part:
    url = block.get("image_url", {}).get("url")
    if not isinstance(url, str) or not url:
        msg = "image_url.url is required for image content"
        raise InvalidRequestError(msg, provider_name=provider_name)

    if url.startswith("data:"):
        mime_type, raw_data = _parse_data_uri(url, "image_url.url", provider_name)
        _validate_inline_size(raw_data, "image_url.url", provider_name)
        return types.Part.from_bytes(data=raw_data, mime_type=mime_type)

    guessed_type, _ = mimetypes.guess_type(url)
    return types.Part.from_uri(file_uri=url, mime_type=guessed_type or "image/jpeg")


def _convert_file_to_part(block: dict[str, Any], provider_name: str) -> types.Part:
    file_data = block.get("file", {}).get("file_data")
    if not isinstance(file_data, str) or not file_data:
        msg = "file.file_data is required for file content"
        raise InvalidRequestError(msg, provider_name=provider_name)

    if file_data.startswith("data:"):
        mime_type, raw_data = _parse_data_uri(file_data, "file.file_data", provider_name)
        _validate_inline_size(raw_data, "file.file_data", provider_name)
        return types.Part.from_bytes(data=raw_data, mime_type=mime_type)

    guessed_type, _ = mimetypes.guess_type(file_data)
    return types.Part.from_uri(file_uri=file_data, mime_type=guessed_type or "application/octet-stream")


def _is_decodable_signature(signature: str) -> bool:
    """Return True if *signature* is base64 the google-genai SDK can decode.

    The SDK reads both the standard and the URL-safe alphabet and tolerates missing
    padding, so match that leniency rather than being stricter than the layer below.
    """
    normalized = signature.replace("-", "+").replace("_", "/")
    try:
        base64.b64decode(normalized + "=" * (-len(normalized) % 4), validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


def _extract_google_thought_signature(container: dict[str, Any], provider_name: str) -> str | bytes | None:
    """Extract the thought_signature stored on a message's or tool call's extra_content.

    A container that carries no Google signature (no extra_content, another provider's
    payload, an explicit None) yields None. A signature that is present but undecodable is
    a caller error, so it is rejected here instead of surfacing from the SDK as a bare
    pydantic ValidationError several frames away.
    """
    extra_content = container.get("extra_content")
    if not isinstance(extra_content, dict) or not isinstance(google_extra := extra_content.get("google"), dict):
        return None

    signature = google_extra.get("thought_signature")
    if signature is None:
        return None
    if isinstance(signature, bytes):
        return signature
    if not isinstance(signature, str) or not signature or not _is_decodable_signature(signature):
        msg = (
            "extra_content['google']['thought_signature'] must be a non-empty base64 string "
            f"or raw bytes, got {signature!r}"
        )
        raise InvalidRequestError(msg, provider_name=provider_name)
    return signature


def _convert_messages(
    messages: list[dict[str, Any]], provider_name: str = "gemini"
) -> tuple[list[types.Content], str | None]:
    """Convert messages to Google GenAI format."""
    formatted_messages = []
    system_instruction = None
    tool_names: dict[str, str] = {}  # tool_call id -> function name, for tool results that carry no name

    for message in messages:
        if message["role"] == "system":
            if system_instruction is None:
                system_instruction = message["content"]
            else:
                system_instruction += f"\n{message['content']}"
        elif message["role"] == "user":
            if isinstance(message["content"], str):
                parts = [types.Part.from_text(text=message["content"])]
            else:
                parts = []
                for content in message["content"]:
                    if content["type"] == "text":
                        parts.append(types.Part.from_text(text=content["text"]))
                    elif content["type"] == "image_url":
                        parts.append(_convert_image_url_to_part(content, provider_name))
                    elif content["type"] == "file":
                        parts.append(_convert_file_to_part(content, provider_name))
                    else:
                        logger.debug("Skipping unsupported Gemini content block type: %s", content.get("type"))
            formatted_messages.append(types.Content(role="user", parts=parts))
        elif message["role"] == "assistant":
            parts = []
            # The model's own text belongs in its turn, ahead of any function calls it made.
            content = message.get("content")
            has_text = isinstance(content, str) and content
            if has_text or not message.get("tool_calls"):
                parts.append(
                    types.Part(
                        text=content,
                        # The SDK field is typed bytes but its validator decodes a base64 str.
                        thought_signature=cast(
                            "bytes | None", _extract_google_thought_signature(message, provider_name)
                        ),
                    )
                )
            if message.get("tool_calls"):
                for i, tool_call in enumerate(message["tool_calls"]):
                    function_call = tool_call["function"]
                    if tool_call_id := tool_call.get("id"):
                        tool_names[tool_call_id] = function_call["name"]
                    arguments = function_call.get("arguments")
                    if isinstance(arguments, (str, bytes, bytearray)) and arguments:
                        try:
                            args = json.loads(arguments)
                        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                            msg = f"Tool call arguments for {function_call['name']!r} must be valid JSON"
                            raise InvalidRequestError(msg, exc, provider_name) from exc
                    else:
                        args = arguments or {}

                    # Extract thought_signature if present (OpenAI compatibility format)
                    # SDK accepts base64 string or bytes
                    thought_signature = _extract_google_thought_signature(tool_call, provider_name)

                    # For the first function call in parallel calls, if no thought_signature is present,
                    # use the skip validator sentinel per Google's documentation:
                    # https://ai.google.dev/gemini-api/docs/thought-signatures#faqs
                    if i == 0 and thought_signature is None:
                        thought_signature = "skip_thought_signature_validator"

                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(name=function_call["name"], args=args),
                            thought_signature=cast("bytes | None", thought_signature),
                        )
                    )

            formatted_messages.append(types.Content(role="model", parts=parts))
        elif message["role"] == "tool":
            name = message.get("name") or tool_names.get(message.get("tool_call_id", ""), "unknown")
            content = message["content"]
            if isinstance(content, (str, bytes, bytearray)):
                with suppress(json.JSONDecodeError, UnicodeDecodeError):
                    content = json.loads(content)
            part = types.Part.from_function_response(name=name, response=_normalize_tool_response(content))
            formatted_messages.append(types.Content(role="function", parts=[part]))

    return formatted_messages, system_instruction


def _normalize_tool_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    return {"result": response}


def _extract_usage_dict(response: types.GenerateContentResponse) -> dict[str, Any]:
    """Extract usage from a Gemini response as a dict.

    Gemini's ``prompt_token_count`` already includes cached tokens
    (``cached_content_token_count`` is a subset).

    Reference: https://ai.google.dev/gemini-api/docs/caching
    """
    metadata = response.usage_metadata
    if metadata is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage: dict[str, Any] = {
        "prompt_tokens": metadata.prompt_token_count or 0,
        # thoughts_token_count is billed output and already counted in total_token_count
        "completion_tokens": (metadata.candidates_token_count or 0) + (metadata.thoughts_token_count or 0),
        "total_tokens": metadata.total_token_count or 0,
    }
    if metadata.thoughts_token_count:
        usage["completion_tokens_details"] = CompletionTokensDetails(reasoning_tokens=metadata.thoughts_token_count)
    if metadata.cached_content_token_count:
        usage["prompt_tokens_details"] = PromptTokensDetails(cached_tokens=metadata.cached_content_token_count)
    return usage


def _thought_signature_extra_content(part: types.Part) -> dict[str, Any] | None:
    """Build the OpenAI-compatible extra_content payload for a Gemini thought_signature, if present.

    Shared by both the non-streaming (_convert_response_to_response_dict) and streaming
    (_create_openai_chunk_from_google_chunk) conversion paths so they stay in sync.
    """
    if part.thought_signature is not None and isinstance(part.thought_signature, bytes):
        return {"google": {"thought_signature": base64.b64encode(part.thought_signature).decode("utf-8")}}
    return None


def _inline_data_image(part: types.Part) -> dict[str, Any] | None:
    """Convert Gemini inline data into an OpenAI-compatible data URL."""
    blob = part.inline_data
    if (
        blob is None
        or not isinstance(blob.data, bytes)
        or not blob.data
        or not isinstance(blob.mime_type, str)
        or not blob.mime_type.startswith("image/")
    ):
        return None
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{blob.mime_type};base64,{base64.b64encode(blob.data).decode('ascii')}"},
    }


def _inline_data_audio(part: types.Part, transcript: str) -> dict[str, Any] | None:
    """Convert Gemini audio inline data into an OpenAI-compatible audio object."""
    blob = part.inline_data
    if (
        blob is None
        or not isinstance(blob.data, bytes)
        or not blob.data
        or not isinstance(blob.mime_type, str)
        or not blob.mime_type.startswith("audio/")
    ):
        return None
    return {
        "id": "google_genai_audio",
        "data": base64.b64encode(blob.data).decode("ascii"),
        "expires_at": 0,
        "transcript": transcript,
    }


_FINISH_REASON_MAP: dict[types.FinishReason, Literal["stop", "length", "content_filter"]] = {
    types.FinishReason.STOP: "stop",
    types.FinishReason.MAX_TOKENS: "length",
    types.FinishReason.SAFETY: "content_filter",
    types.FinishReason.RECITATION: "content_filter",
    types.FinishReason.PROHIBITED_CONTENT: "content_filter",
    types.FinishReason.BLOCKLIST: "content_filter",
    types.FinishReason.SPII: "content_filter",
    types.FinishReason.IMAGE_SAFETY: "content_filter",
    types.FinishReason.IMAGE_PROHIBITED_CONTENT: "content_filter",
    types.FinishReason.IMAGE_RECITATION: "content_filter",
}


def _map_finish_reason(
    finish_reason: types.FinishReason | None,
) -> Literal["stop", "length", "content_filter"] | None:
    """Map a Gemini finish reason onto the OpenAI vocabulary.

    Returns None for reasons without an OpenAI counterpart so that each call site can
    choose its own fallback: a terminal "stop" for complete responses, or None for
    stream chunks, where forcing a terminal reason would mislabel non-final chunks.
    """
    return _FINISH_REASON_MAP.get(finish_reason) if finish_reason is not None else None


def _resolve_finish_reason(
    mapped_finish_reason: Literal["stop", "length", "content_filter"] | None,
    has_tool_calls: bool,
) -> Literal["stop", "length", "content_filter", "tool_calls"] | None:
    """Combine the mapped finish reason with tool call presence.

    Truncation and filtering take priority over "tool_calls": a response cut short mid
    tool call must not look like a completed tool call round.
    """
    if has_tool_calls and mapped_finish_reason not in ("length", "content_filter"):
        return "tool_calls"
    return mapped_finish_reason


def _prompt_was_blocked(response: types.GenerateContentResponse) -> bool:
    """Return whether Gemini rejected the prompt before producing a candidate."""
    prompt_feedback = response.prompt_feedback
    return (
        prompt_feedback is not None
        and isinstance(prompt_feedback.block_reason, types.BlockedReason)
        and prompt_feedback.block_reason is not types.BlockedReason.BLOCKED_REASON_UNSPECIFIED
    )


def _convert_response_to_response_dict(response: types.GenerateContentResponse) -> dict[str, Any]:
    """Convert a Gemini GenerateContentResponse into an OpenAI-shaped completion dict."""
    response_dict = {
        "id": "google_genai_response",
        "model": "google/genai",
        "created": 0,
        "usage": _extract_usage_dict(response),
    }

    choices: list[dict[str, Any]] = []
    if response.candidates:
        candidate = response.candidates[0]
        mapped_finish_reason = _map_finish_reason(candidate.finish_reason)

        reasoning = None
        tool_calls_list: list[dict[str, Any]] = []
        text_content = None
        images: list[dict[str, Any]] = []
        audio_part: types.Part | None = None
        # Gemini 3 signs the last non-function-call part of a text answer. It rides message.extra_content,
        # the same spelling Google's OpenAI-compatible endpoint uses.
        message_extra_content = None
        parts = candidate.content.parts if candidate.content else None

        for part in parts or []:
            if part.thought:
                reasoning = (reasoning or "") + (part.text or "")
            elif function_call := part.function_call:
                args_dict = {}
                if args := getattr(function_call, "args", None):
                    for key, value in args.items():
                        args_dict[key] = value

                tool_call_dict: dict[str, Any] = {
                    "id": f"call_{hash(function_call.name)}_{len(tool_calls_list)}",
                    "function": {
                        "name": function_call.name,
                        "arguments": json.dumps(args_dict),
                    },
                    "type": "function",
                }

                # Include thought_signature if present (OpenAI compatibility format)
                if extra_content := _thought_signature_extra_content(part):
                    tool_call_dict["extra_content"] = extra_content

                tool_calls_list.append(tool_call_dict)
            else:
                if image := _inline_data_image(part):
                    images.append(image)
                if (
                    part.inline_data
                    and isinstance(part.inline_data.mime_type, str)
                    and part.inline_data.mime_type.startswith("audio/")
                ):
                    audio_part = part
                if part.text:
                    text_content = (text_content or "") + part.text
                message_extra_content = _thought_signature_extra_content(part) or message_extra_content

        audio = _inline_data_audio(audio_part, text_content or "") if audio_part else None

        # Truncated or filtered responses produce a choice even without content or tool
        # calls, e.g. a thinking model that spent the whole max_output_tokens budget on
        # reasoning, so callers see the terminal reason instead of an empty choices list.
        if tool_calls_list or text_content or images or audio or mapped_finish_reason in ("length", "content_filter"):
            choices.append(
                {
                    "message": {
                        "role": "assistant",
                        "content": text_content,
                        "reasoning": reasoning or None,
                        "tool_calls": tool_calls_list or None,
                        "images": images or None,
                        "audio": audio,
                        "extra_content": message_extra_content,
                        "refusal": _GEMINI_CONTENT_FILTER_REFUSAL if mapped_finish_reason == "content_filter" else None,
                    },
                    "finish_reason": _resolve_finish_reason(mapped_finish_reason, bool(tool_calls_list)) or "stop",
                    "index": 0,
                }
            )
    elif _prompt_was_blocked(response):
        choices.append(
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": None,
                    "tool_calls": None,
                    "extra_content": None,
                    "refusal": _GEMINI_CONTENT_FILTER_REFUSAL,
                },
                "finish_reason": "content_filter",
                "index": 0,
            }
        )

    response_dict["choices"] = choices

    return response_dict


def _create_openai_embedding_response_from_google(
    model: str, result: types.EmbedContentResponse
) -> CreateEmbeddingResponse:
    """Convert a Google embedding response to an OpenAI-compatible format."""

    data = [
        Embedding(
            embedding=embedding.values,
            index=i,
            object="embedding",
        )
        for i, embedding in enumerate(result.embeddings or [])
        if embedding.values
    ]

    usage = Usage(prompt_tokens=0, total_tokens=0)

    return CreateEmbeddingResponse(
        data=data,
        model=model,
        object="list",
        usage=usage,
    )


def _create_openai_chunk_from_google_chunk(
    response: types.GenerateContentResponse,
    tool_call_counter: list[int] | None = None,
) -> ChatCompletionChunk:
    """Convert a Google GenerateContentResponse to an OpenAI ChatCompletionChunk.

    Args:
        response: The Google GenerateContentResponse streaming chunk.
        tool_call_counter: Optional single-element list holding the number of tool
            calls already emitted earlier in the same stream. This should be created
            once per stream and passed in by the caller so that ``index`` (and the
            generated tool call ``id``) stay stable and unique across chunks, rather
            than restarting at 0 for every chunk.
    """

    candidate = response.candidates[0] if response.candidates else None

    if tool_call_counter is None:
        tool_call_counter = [0]

    content = ""
    reasoning_content = ""
    tool_calls_list: list[ChoiceDeltaToolCall] = []
    message_extra_content = None
    images: list[dict[str, Any]] = []
    audio_part: types.Part | None = None

    # Content can be absent on terminal chunks, e.g. when the response is truncated or
    # filtered before any part is produced; the finish reason must still be surfaced.
    parts = candidate.content.parts if candidate and candidate.content else None

    for part in parts or []:
        if part.thought:
            reasoning_content += part.text or ""
        elif function_call := part.function_call:
            args_dict = {}
            if args := function_call.args:
                for key, value in args.items():
                    args_dict[key] = value

            # Include thought_signature if present (OpenAI compatibility format), mirroring
            # the non-streaming conversion in _convert_response_to_response_dict.
            extra_content = _thought_signature_extra_content(part)

            tool_call_index = tool_call_counter[0]
            tool_call_counter[0] += 1

            tool_calls_list.append(
                ChoiceDeltaToolCall(
                    index=tool_call_index,
                    id=f"call_{hash(function_call.name)}_{tool_call_index}",
                    type="function",
                    function=ChoiceDeltaToolCallFunction(
                        name=function_call.name,
                        arguments=json.dumps(args_dict),
                    ),
                    extra_content=extra_content,
                )
            )
        else:
            if image := _inline_data_image(part):
                images.append(image)
            if (
                part.inline_data
                and isinstance(part.inline_data.mime_type, str)
                and part.inline_data.mime_type.startswith("audio/")
            ):
                audio_part = part
            content += part.text or ""  # the signed final part may carry empty text
            message_extra_content = _thought_signature_extra_content(part) or message_extra_content

    audio = None
    if audio_part and (converted_audio := _inline_data_audio(audio_part, content)):
        audio = AudioContent(**converted_audio)

    # Unmapped reasons stay None so non-final chunks are not forced to a terminal reason.
    mapped_finish_reason = _map_finish_reason(candidate.finish_reason) if candidate else None
    prompt_was_blocked = _prompt_was_blocked(response)
    finish_reason = (
        "content_filter" if prompt_was_blocked else _resolve_finish_reason(mapped_finish_reason, bool(tool_calls_list))
    )

    delta = ChoiceDelta(
        content=content or None,
        role="assistant",
        refusal=_GEMINI_CONTENT_FILTER_REFUSAL if finish_reason == "content_filter" else None,
        reasoning=Reasoning(content=reasoning_content) if reasoning_content else None,
        tool_calls=tool_calls_list or None,
        extra_content=message_extra_content,
        images=cast("list[ImageContent] | None", images or None),
        audio=audio,
    )

    choice = ChunkChoice(
        index=0,
        delta=delta,
        finish_reason=finish_reason,
    )

    usage = None
    if response.usage_metadata:
        cached_tokens = response.usage_metadata.cached_content_token_count
        thought_tokens = response.usage_metadata.thoughts_token_count
        usage = CompletionUsage(
            prompt_tokens=response.usage_metadata.prompt_token_count or 0,
            completion_tokens=(response.usage_metadata.candidates_token_count or 0) + (thought_tokens or 0),
            total_tokens=response.usage_metadata.total_token_count or 0,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens) if cached_tokens else None,
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=thought_tokens)
            if thought_tokens
            else None,
        )

    return ChatCompletionChunk(
        id=f"chatcmpl-{time()}",
        choices=[choice],
        created=int(time()),
        model=str(response.model_version),
        object="chat.completion.chunk",
        usage=usage,
    )


def _convert_models_list(models_list: Pager[types.Model]) -> list[Model]:
    return [Model(id=model.name or "Unknown", object="model", created=0, owned_by="google") for model in models_list]


_GOOGLE_TO_OPENAI_STATUS_MAP: dict[str, str] = {
    "JOB_STATE_QUEUED": "validating",
    "JOB_STATE_PENDING": "validating",
    "JOB_STATE_RUNNING": "in_progress",
    "JOB_STATE_PAUSED": "in_progress",
    "JOB_STATE_UPDATING": "in_progress",
    "JOB_STATE_SUCCEEDED": "completed",
    "JOB_STATE_PARTIALLY_SUCCEEDED": "completed",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_CANCELLING": "cancelling",
    "JOB_STATE_CANCELLED": "cancelled",
    "JOB_STATE_EXPIRED": "expired",
}


def _convert_google_batch_job_to_openai_batch(job: types.BatchJob) -> Batch:
    """Convert a Google GenAI ``BatchJob`` to an OpenAI ``Batch``."""
    state_str = job.state.value if job.state else "JOB_STATE_UNSPECIFIED"
    openai_status = _GOOGLE_TO_OPENAI_STATUS_MAP.get(state_str)
    if openai_status is None:
        logger.warning("Unknown Google batch state: %s, defaulting to 'in_progress'", state_str)
        openai_status = "in_progress"

    stats = job.completion_stats
    total = 0
    completed = 0
    failed = 0
    if stats:
        successful = stats.successful_count or 0
        failed_count = stats.failed_count or 0
        incomplete = stats.incomplete_count or 0
        total = successful + failed_count + incomplete
        completed = successful
        failed = failed_count

    request_counts = BatchRequestCounts(total=total, completed=completed, failed=failed)

    created_at = int(job.create_time.timestamp()) if job.create_time else 0

    output_uri = None
    if job.output_info:
        output_uri = job.output_info.gcs_output_directory or job.output_info.bigquery_output_table

    src_uri = ""
    if job.src:
        if job.src.gcs_uri:
            src_uri = job.src.gcs_uri[0] if job.src.gcs_uri else ""
        elif job.src.file_name:
            src_uri = job.src.file_name

    metadata: dict[str, str] | None = None
    if job.display_name:
        metadata = {"displayName": job.display_name}

    return Batch(
        id=job.name or "",
        object="batch",
        endpoint="/v1/chat/completions",
        status=cast("Any", openai_status),
        created_at=created_at,
        completion_window="24h",
        request_counts=request_counts,
        input_file_id=src_uri,
        output_file_id=output_uri,
        error_file_id=None,
        metadata=metadata,
    )


def _convert_openai_request_to_inlined_request(
    entry: dict[str, Any],
    provider_name: str = "gemini",
) -> types.InlinedRequest:
    """Convert a single OpenAI-format JSONL entry to a Google ``InlinedRequest``.

    The *entry* dict follows the OpenAI batch format with ``custom_id`` and
    ``body`` (containing ``model``, ``messages``, and optional parameters).
    """
    body = entry.get("body", {})
    messages = body.get("messages", [])
    model = body.get("model", "")

    contents, system_instruction = _convert_messages(messages, provider_name=provider_name)

    config_kwargs: dict[str, Any] = {}
    if "max_tokens" in body:
        config_kwargs["max_output_tokens"] = body["max_tokens"]
    if "temperature" in body:
        config_kwargs["temperature"] = body["temperature"]
    if "top_p" in body:
        config_kwargs["top_p"] = body["top_p"]
    if "stop" in body:
        stop = body["stop"]
        config_kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    metadata: dict[str, str] | None = None
    custom_id = entry.get("custom_id")
    if custom_id:
        metadata = {"custom_id": str(custom_id)}

    return types.InlinedRequest(
        model=model,
        contents=cast("Any", contents),
        config=config,
        metadata=metadata,
    )


def _convert_google_batch_output_to_result(output_lines: list[str]) -> BatchResult:
    """Parse Google batch output JSONL lines into a ``BatchResult``.

    Each output line is a JSON object containing the prediction result from the
    Gemini/Vertex AI batch inference job.
    """
    from any_llm.providers.gemini.base import GoogleProvider

    results: list[BatchResultItem] = []
    for line in output_lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        custom_id = ""
        if "request" in record and "metadata" in record["request"]:
            custom_id = record["request"]["metadata"].get("custom_id", "")

        item = BatchResultItem(custom_id=custom_id)

        response_data = record.get("response")
        if response_data is None:
            error_data = record.get("error")
            if error_data:
                item.error = BatchResultError(
                    code=str(error_data.get("code", "unknown")),
                    message=error_data.get("message", "Unknown error"),
                )
            else:
                item.error = BatchResultError(
                    code="unknown",
                    message="Record contains neither response nor error",
                )
        else:
            try:
                response = types.GenerateContentResponse.model_validate(response_data)
                response_dict = _convert_response_to_response_dict(response)
                item.result = GoogleProvider._convert_completion_response((response_dict, record.get("model", "")))
            except Exception as exc:
                item.error = BatchResultError(
                    code="parse_error",
                    message=f"Failed to parse response: {exc}",
                )

        results.append(item)

    return BatchResult(results=results)
