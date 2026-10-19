import base64
import binascii
import json
import mimetypes
import re
from bisect import bisect_right
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from time import time
from typing import Any, Literal, cast
from uuid import uuid4

from google.genai import types
from google.genai.pagers import Pager
from pydantic import ValidationError

from any_llm.exceptions import InvalidRequestError, ProviderError, UnsupportedParameterError
from any_llm.logging import logger
from any_llm.types.batch import Batch, BatchRequestCounts, BatchResult, BatchResultError, BatchResultItem
from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaAudio,
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
# https://ai.google.dev/gemini-api/docs/function-calling#multimodal
_FUNCTION_RESPONSE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "application/pdf", "text/plain"})
_GEMINI_MAJOR_VERSION = re.compile(r"(?:^|/)gemini-(\d+)")
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
    Gemini API rejects with a 400. Such schemas must be routed as raw JSON Schema:
    ``GenerateContentConfig.response_json_schema`` for a response, and
    ``FunctionDeclaration.parameters_json_schema`` for a tool's parameters.
    """
    if isinstance(schema, dict):
        if "additionalProperties" in schema:
            return True
        return any(_has_additional_properties(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_additional_properties(v) for v in schema)
    return False


def _raw_function_declaration(function: dict[str, Any], params: dict[str, Any]) -> types.FunctionDeclaration:
    """A declaration whose parameters the SDK forwards to the server as raw JSON Schema."""
    return types.FunctionDeclaration(
        name=function["name"],
        description=function.get("description", ""),
        parameters_json_schema=params,
    )


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
            function_declarations.append(_raw_function_declaration(function, params))
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

        # An ``items`` schema is copied verbatim, so it can carry valid JSON Schema that
        # ``types.Schema`` does not model (``exclusiveMinimum``, ``const``, ...) or misserializes
        # (``additionalProperties``). Such a schema goes out raw, as one with a ``$ref`` does. Any
        # other validation error is the schema's own and is refused here, before the request.
        if any(_has_additional_properties(prop.get("items")) for prop in properties.values()):
            function_declarations.append(_raw_function_declaration(function, params))
            continue
        try:
            parameters = types.Schema(**parameters_dict)
        except ValidationError as exc:
            if any(error["type"] != "extra_forbidden" for error in exc.errors()):
                msg = f"Tool {function['name']!r} has an invalid parameters schema: {exc.errors()[0]['msg']}"
                raise InvalidRequestError(msg, provider_name=provider_name) from exc
            function_declarations.append(_raw_function_declaration(function, params))
            continue

        function_declarations.append(
            types.FunctionDeclaration(
                name=function["name"],
                description=function.get("description", ""),
                parameters=parameters,
            )
        )
    if function_declarations:
        converted_tools.append(types.Tool(function_declarations=function_declarations))
    return converted_tools


def _enable_server_side_tool_invocations(
    tools: list[types.Tool], tool_config: types.ToolConfig | None
) -> types.ToolConfig | None:
    """Return ``tool_config`` with server-side tool invocations on when built-in tools sit beside functions.

    Gemini answers a request that combines built-in tools (such as ``google_search`` or ``code_execution``)
    with function declarations with a 400 unless ``include_server_side_tool_invocations`` is set.
    """
    has_functions = any(tool.function_declarations for tool in tools)
    has_built_in = any(tool.model_dump(exclude_none=True, exclude={"function_declarations"}) for tool in tools)
    if not (has_functions and has_built_in):
        return tool_config
    return (tool_config or types.ToolConfig()).model_copy(update={"include_server_side_tool_invocations": True})


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

    return mime_type, _decode_base64(data_uri.split("base64,", 1)[1], field_name, provider_name)


def _decode_base64(encoded_data: str, field_name: str, provider_name: str) -> bytes:
    """Decode strict base64 within the inline upload limit, reporting bad input as an invalid request.

    The decoded size is computed from the encoded length and padding, so an oversized payload is
    rejected before anything is decoded.
    """
    if not encoded_data:
        msg = f"{field_name} is missing base64 data"
        raise InvalidRequestError(msg, provider_name=provider_name)
    decoded_size = len(encoded_data) * 3 // 4 - encoded_data[-2:].count("=")
    if decoded_size > _INLINE_SIZE_LIMIT:
        msg = f"{field_name} exceeds the 20 MB inline upload limit for {provider_name} ({decoded_size} bytes)"
        raise InvalidRequestError(msg, provider_name=provider_name)

    try:
        return base64.b64decode(encoded_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        msg = f"{field_name} contains invalid base64 data"
        raise InvalidRequestError(msg, exc, provider_name) from exc


def _convert_image_url_to_part(block: dict[str, Any], provider_name: str) -> types.Part:
    url = block.get("image_url", {}).get("url")
    if not isinstance(url, str) or not url:
        msg = "image_url.url is required for image content"
        raise InvalidRequestError(msg, provider_name=provider_name)

    if url.startswith("data:"):
        mime_type, raw_data = _parse_data_uri(url, "image_url.url", provider_name)
        return types.Part.from_bytes(data=raw_data, mime_type=mime_type)

    guessed_type, _ = mimetypes.guess_type(url)
    return types.Part.from_uri(file_uri=url, mime_type=guessed_type or "image/jpeg")


def _convert_input_audio_to_part(block: dict[str, Any], provider_name: str) -> types.Part:
    """OpenAI's input_audio part: base64 data plus a format name, which Gemini spells as audio/<format>."""
    audio = block.get("input_audio")
    if not isinstance(audio, dict):
        audio = {}
    data = audio.get("data")
    audio_format = audio.get("format")
    if not isinstance(data, str) or not isinstance(audio_format, str) or not audio_format:
        msg = "input_audio.data and input_audio.format are required for audio content"
        raise InvalidRequestError(msg, provider_name=provider_name)

    raw_data = _decode_base64(data, "input_audio.data", provider_name)
    return types.Part.from_bytes(data=raw_data, mime_type=f"audio/{audio_format.lower()}")


def _convert_file_to_part(block: dict[str, Any], provider_name: str) -> types.Part:
    file_data = block.get("file", {}).get("file_data")
    if not isinstance(file_data, str) or not file_data:
        msg = "file.file_data is required for file content"
        raise InvalidRequestError(msg, provider_name=provider_name)

    if file_data.startswith("data:"):
        mime_type, raw_data = _parse_data_uri(file_data, "file.file_data", provider_name)
        return types.Part.from_bytes(data=raw_data, mime_type=mime_type)

    # A provider-hosted file URI carries no extension, so the block's filename is the only
    # MIME hint available for a file referenced by URI, including one uploaded through the Files API.
    filename = block.get("file", {}).get("filename")
    guessed_type = mimetypes.guess_type(filename)[0] if isinstance(filename, str) and filename else None
    if guessed_type is None:
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


_SYNTHETIC_CODE_EXECUTION_ID_PREFIX = "code_exec_"
_SYNTHETIC_TOOL_CALL_ID_PREFIX = "call_anyllm_"


def _tool_call_id(function_call: types.FunctionCall, response_id: str | None, index: int) -> str:
    """The id Gemini issued for a function call, else one minted from the response so converting it again repeats it."""
    if function_call.id:
        return function_call.id
    if response_id:
        return f"{_SYNTHETIC_TOOL_CALL_ID_PREFIX}{response_id}_{index}"
    return f"{_SYNTHETIC_TOOL_CALL_ID_PREFIX}{uuid4().hex}"


@dataclass
class CodeExecutionState:
    """Pairs code execution results and outputs with their code across the parts of one response or stream."""

    executable_code_count: int = 0
    last_executable_code_id: str | None = None
    # The code or result that the next inline output (a chart, a file) was produced by.
    last_output_owner_id: str | None = None

    def next_id(self) -> str:
        return f"{_SYNTHETIC_CODE_EXECUTION_ID_PREFIX}{self.executable_code_count}"


def _enum_value(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value.value) if isinstance(value, Enum) else str(value)


def _code_execution_item(part: types.Part, state: CodeExecutionState) -> dict[str, Any] | None:
    """Map a code execution part, or inline data produced after code ran, to its extra_content item."""
    if isinstance(executable_code := part.executable_code, types.ExecutableCode):
        code_id = executable_code.id or state.next_id()
        state.executable_code_count += 1
        state.last_executable_code_id = code_id
        state.last_output_owner_id = code_id
        return {
            "type": "executable_code",
            "id": code_id,
            "language": _enum_value(executable_code.language, "LANGUAGE_UNSPECIFIED"),
            "code": executable_code.code or "",
        }
    if isinstance(result := part.code_execution_result, types.CodeExecutionResult):
        result_id = result.id or state.last_executable_code_id or state.next_id()
        if state.executable_code_count:
            state.last_output_owner_id = result_id
        return {
            "type": "code_execution_result",
            "id": result_id,
            "outcome": _enum_value(result.outcome, "OUTCOME_UNSPECIFIED"),
            "output": result.output or "",
        }
    blob = part.inline_data
    if state.last_output_owner_id is not None and isinstance(blob, types.Blob) and isinstance(blob.data, bytes):
        return {
            "type": "code_execution_output",
            "id": state.last_output_owner_id,
            "mime_type": blob.mime_type or "",
            "data": base64.b64encode(blob.data).decode("ascii"),
        }
    return None


def _merge_code_execution(
    extra_content: dict[str, Any] | None, code_execution: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Add code execution items under extra_content["google"], beside any thought_signature."""
    if not code_execution:
        return extra_content
    merged = dict(extra_content or {})
    merged["google"] = {**merged.get("google", {}), "code_execution": code_execution}
    return merged


def _code_execution_parts(message: dict[str, Any]) -> list[types.Part]:
    """Rebuild the code execution parts an assistant message carries in extra_content, skipping malformed items."""
    extra_content = message.get("extra_content")
    if not isinstance(extra_content, dict) or not isinstance(google_extra := extra_content.get("google"), dict):
        return []
    items = google_extra.get("code_execution")
    if not isinstance(items, list):
        return []
    parts: list[types.Part] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        # A synthesized id was never issued by Gemini, so it is not sent back.
        if not isinstance(item_id, str) or item_id.startswith(_SYNTHETIC_CODE_EXECUTION_ID_PREFIX):
            item_id = None
        try:
            if item.get("type") == "executable_code":
                parts.append(
                    types.Part(
                        executable_code=types.ExecutableCode(
                            code=item.get("code"), language=item.get("language"), id=item_id
                        )
                    )
                )
            elif item.get("type") == "code_execution_result":
                parts.append(
                    types.Part(
                        code_execution_result=types.CodeExecutionResult(
                            outcome=item.get("outcome"), output=item.get("output"), id=item_id
                        )
                    )
                )
        except ValidationError:
            logger.debug("Skipping malformed Gemini code execution item: %r", item)
    return parts


def _pending_function_response_parts(contents: list[types.Content]) -> list[types.Part] | None:
    """Return the parts of a trailing user turn made only of function responses, else None."""
    if not contents:
        return None
    last = contents[-1]
    if last.role != "user" or not last.parts:
        return None
    if any(part.function_response is None for part in last.parts):
        return None
    return last.parts


def _system_text(content: Any) -> str:
    """The text of a system message, whose content may be a string or a list of text parts."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")


def _supports_multimodal_function_responses(model: str | None) -> bool:
    """Whether the model reads media parts of a function response; Gemini 2.5 and older reject them with a 400."""
    match = _GEMINI_MAJOR_VERSION.search(model or "")
    return match is not None and int(match.group(1)) >= 3


def _convert_messages(
    messages: list[dict[str, Any]], provider_name: str = "gemini", model: str | None = None
) -> tuple[list[types.Content], str | None]:
    """Convert messages to Google GenAI format.

    ``model`` decides whether media in tool results is sent as function response parts.
    """
    multimodal_tool_results = _supports_multimodal_function_responses(model)
    formatted_messages = []
    system_instruction = None
    tool_names: dict[str, str] = {}  # tool_call id -> function name, for tool results that carry no name
    # An id any-llm minted was never issued by Gemini, and one shared by several calls (as ids from
    # older any-llm versions are) pairs ambiguously, so neither is sent back.
    id_counts = Counter(
        tool_call.get("id")
        for message in messages
        if message["role"] == "assistant"
        for tool_call in message.get("tool_calls") or []
    )
    call_order: dict[str, int] = {}  # tool_call id -> position among the latest model turn's calls
    sent_call_ids: set[str] = set()  # ids sent on the latest model turn's calls
    pending_order: list[int] = []  # call positions of the results in the pending user turn

    for message in messages:
        if message["role"] in ("system", "developer"):
            # Gemini has a single system_instruction; OpenAI's developer role is the same instruction channel.
            text = _system_text(message["content"])
            system_instruction = text if system_instruction is None else f"{system_instruction}\n{text}"
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
                    elif content["type"] == "input_audio":
                        parts.append(_convert_input_audio_to_part(content, provider_name))
                    else:
                        logger.debug("Skipping unsupported Gemini content block type: %s", content.get("type"))
            formatted_messages.append(types.Content(role="user", parts=parts))
        elif message["role"] == "assistant":
            parts = _code_execution_parts(message)
            # The model's own text belongs in its turn, ahead of any function calls it made.
            content = message.get("content")
            has_text = isinstance(content, str) and content
            # Code execution parts already give the turn content, so it needs no empty text part.
            if has_text or not (parts or message.get("tool_calls")):
                parts.append(
                    types.Part(
                        text=content,
                        # The SDK field is typed bytes but its validator decodes a base64 str.
                        thought_signature=cast(
                            "bytes | None", _extract_google_thought_signature(message, provider_name)
                        ),
                    )
                )
            call_order = {}
            sent_call_ids = set()
            if message.get("tool_calls"):
                for i, tool_call in enumerate(message["tool_calls"]):
                    function_call = tool_call["function"]
                    sent_id = None
                    if tool_call_id := tool_call.get("id"):
                        tool_names[tool_call_id] = function_call["name"]
                        call_order.setdefault(tool_call_id, i)
                        if id_counts[tool_call_id] == 1 and not tool_call_id.startswith(_SYNTHETIC_TOOL_CALL_ID_PREFIX):
                            sent_id = tool_call_id
                            sent_call_ids.add(tool_call_id)
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
                            function_call=types.FunctionCall(id=sent_id, name=function_call["name"], args=args),
                            thought_signature=cast("bytes | None", thought_signature),
                        )
                    )

            formatted_messages.append(types.Content(role="model", parts=parts))
        elif message["role"] == "tool":
            tool_call_id = message.get("tool_call_id") or ""
            name = message.get("name") or tool_names.get(tool_call_id, "unknown")
            content = message["content"]
            if isinstance(content, (str, bytes, bytearray)):
                with suppress(json.JSONDecodeError, UnicodeDecodeError):
                    content = json.loads(content)
            media: list[types.FunctionResponsePart] = []
            if multimodal_tool_results and isinstance(content, list):
                content, media = _extract_function_response_media(content, provider_name)
            # Gemini 3 maps each result to its call by id; an id matching no call it was sent would not pair.
            part = types.Part(
                function_response=types.FunctionResponse(
                    id=tool_call_id if tool_call_id in sent_call_ids else None,
                    name=name,
                    response=_normalize_tool_response(content),
                    parts=media or None,
                )
            )
            position = call_order.get(tool_call_id, len(call_order))
            # The API documents only "user" and "model" as Content roles. "function" is tolerated by
            # older models and rejected with 400 INVALID_ARGUMENT by gemini-3.5-flash-lite, 3.7-flash
            # and 3.8-flash. Results of parallel calls share one user turn, as google-genai emits them.
            if (pending := _pending_function_response_parts(formatted_messages)) is not None:
                # Models that issue no ids pair results with calls by position, so results go in call order.
                index = bisect_right(pending_order, position)
                pending.insert(index, part)
                pending_order.insert(index, position)
            else:
                formatted_messages.append(types.Content(role="user", parts=[part]))
                pending_order = [position]

    if not formatted_messages and isinstance(system_instruction, str):
        # Gemini refuses a request with no contents ("contents are required"), and OpenAI-style
        # callers send a lone system message as the whole prompt, so it becomes the user's turn.
        return [types.Content(role="user", parts=[types.Part.from_text(text=system_instruction)])], None

    return formatted_messages, system_instruction


def _extract_function_response_media(
    content: list[Any], provider_name: str
) -> tuple[list[Any], list[types.FunctionResponsePart]]:
    """Move data-URI media the API reads in a function response into parts, leaving a ``$ref`` in its place.

    Inside the JSON response such media is only base64 text to the model. The ``$ref`` keeps each
    item's position relative to the text around it. Anything else, including media given by URL
    (which the API drops from a function response), stays in the response unchanged.
    """
    response: list[Any] = []
    media: list[types.FunctionResponsePart] = []
    for block in content:
        data_uri = _tool_block_data_uri(block)
        if data_uri is None or _data_uri_mime_type(data_uri) not in _FUNCTION_RESPONSE_MIME_TYPES:
            response.append(block)
            continue
        field_name = "image_url.url" if block["type"] == "image_url" else "file.file_data"
        mime_type, raw_data = _parse_data_uri(data_uri, field_name, provider_name)
        display_name = f"attachment-{len(media) + 1}"
        media.append(
            types.FunctionResponsePart(
                inline_data=types.FunctionResponseBlob(data=raw_data, mime_type=mime_type, display_name=display_name)
            )
        )
        response.append({"$ref": display_name})
    return response, media


def _tool_block_data_uri(block: Any) -> str | None:
    if not isinstance(block, dict):
        return None
    if block.get("type") == "image_url":
        source, key = block.get("image_url"), "url"
    elif block.get("type") == "file":
        source, key = block.get("file"), "file_data"
    else:
        return None
    value = source.get(key) if isinstance(source, dict) else None
    return value if isinstance(value, str) and value.startswith("data:") else None


def _data_uri_mime_type(data_uri: str) -> str:
    return data_uri[5:].split(",", 1)[0].split(";", 1)[0].lower()


def _normalize_tool_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    return {"result": response}


def _tokens_by_modality(*count_lists: list[types.ModalityTokenCount] | None) -> dict[types.MediaModality, int]:
    totals: dict[types.MediaModality, int] = {}
    for counts in count_lists:
        for count in counts or []:
            if count.modality is not None and count.token_count:
                totals[count.modality] = totals.get(count.modality, 0) + count.token_count
    return totals


def _extract_usage_dict(response: types.GenerateContentResponse) -> dict[str, Any]:
    """Extract Gemini usage using OpenAI-compatible inclusive token counts.

    Gemini's ``prompt_token_count`` already includes cached tokens
    (``cached_content_token_count`` is a subset). Gemini reports tool-use prompt
    tokens and thinking tokens separately, while OpenAI includes those categories
    in ``prompt_tokens`` and ``completion_tokens`` respectively. Audio and image
    output tokens stay inside ``completion_tokens`` and are broken out in its details;
    audio input tokens are broken out of ``prompt_tokens`` the same way, since Gemini
    prices audio input above text.

    Reference: https://ai.google.dev/api/generate-content#UsageMetadata
    """
    metadata = response.usage_metadata
    if metadata is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage: dict[str, Any] = {
        "prompt_tokens": (metadata.prompt_token_count or 0) + (metadata.tool_use_prompt_token_count or 0),
        "completion_tokens": (metadata.candidates_token_count or 0) + (metadata.thoughts_token_count or 0),
        "total_tokens": metadata.total_token_count or 0,
    }
    output_modalities = _tokens_by_modality(metadata.candidates_tokens_details)
    audio_tokens = output_modalities.get(types.MediaModality.AUDIO)
    image_tokens = output_modalities.get(types.MediaModality.IMAGE)
    if metadata.thoughts_token_count or audio_tokens or image_tokens:
        usage["completion_tokens_details"] = CompletionTokensDetails(
            reasoning_tokens=metadata.thoughts_token_count, audio_tokens=audio_tokens, image_tokens=image_tokens
        )
    input_audio_tokens = _tokens_by_modality(
        metadata.prompt_tokens_details, metadata.tool_use_prompt_tokens_details
    ).get(types.MediaModality.AUDIO)
    if metadata.cached_content_token_count or input_audio_tokens:
        usage["prompt_tokens_details"] = PromptTokensDetails(
            cached_tokens=metadata.cached_content_token_count, audio_tokens=input_audio_tokens
        )
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


def _inline_audio_blob(part: types.Part) -> types.Blob | None:
    """Return the part's non-empty audio blob, if it has one."""
    blob = part.inline_data
    if (
        blob is None
        or not isinstance(blob.data, bytes)
        or not blob.data
        or not isinstance(blob.mime_type, str)
        or not blob.mime_type.startswith("audio/")
    ):
        return None
    return blob


def _wav_from_pcm(pcm: bytes, mime_type: str) -> bytes:
    """Wrap headerless 16-bit mono PCM in a WAV header."""
    rate = 24000
    for parameter in mime_type.split(";"):
        parameter = parameter.strip()
        if parameter.startswith("rate="):
            with suppress(ValueError):
                parsed_rate = int(parameter.removeprefix("rate="))
                if parsed_rate > 0:
                    rate = parsed_rate
            break
    return (
        b"RIFF"
        + (36 + len(pcm)).to_bytes(4, "little")
        + b"WAVE"
        + b"fmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + rate.to_bytes(4, "little")
        + (rate * 2).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + len(pcm).to_bytes(4, "little")
        + pcm
    )


def _inline_data_audio(blobs: list[types.Blob], transcript: str, *, playable: bool) -> dict[str, Any] | None:
    """Convert Gemini audio blobs into an OpenAI-compatible audio object.

    Complete audio/L16 responses are wrapped as WAV so the sample rate from the MIME
    type is retained. Streaming chunks stay raw because their total length is unknown.
    """
    if not blobs:
        return None
    data = b"".join(cast("bytes", blob.data) for blob in blobs)
    mime_type = cast("str", blobs[0].mime_type)
    if playable and mime_type.startswith("audio/L16"):
        data = _wav_from_pcm(data, mime_type)
    return {
        "id": "google_genai_audio",
        "data": base64.b64encode(data).decode("ascii"),
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


def _convert_response_to_response_dict(
    response: types.GenerateContentResponse, provider_name: str | None = None
) -> dict[str, Any]:
    """Convert a Gemini GenerateContentResponse into an OpenAI-shaped completion dict.

    Raises:
        ProviderError: If no candidates were returned and the prompt was not blocked.

    """
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
        audio_blobs: list[types.Blob] = []
        # Gemini 3 signs the last non-function-call part of a text answer. It rides message.extra_content,
        # the same spelling Google's OpenAI-compatible endpoint uses.
        message_extra_content = None
        code_execution: list[dict[str, Any]] = []
        code_execution_state = CodeExecutionState()
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
                    "id": _tool_call_id(function_call, response.response_id, len(tool_calls_list)),
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
                if audio_blob := _inline_audio_blob(part):
                    audio_blobs.append(audio_blob)
                if code_execution_item := _code_execution_item(part, code_execution_state):
                    code_execution.append(code_execution_item)
                # An empty text part is still an answer: Gemini 3 answers a conversation
                # that ends on an assistant turn with "" and STOP.
                if part.text is not None:
                    text_content = (text_content or "") + part.text
                message_extra_content = _thought_signature_extra_content(part) or message_extra_content

        message_extra_content = _merge_code_execution(message_extra_content, code_execution)
        audio = _inline_data_audio(audio_blobs, text_content or "", playable=True)

        # Truncated or filtered responses produce a choice even without content or tool
        # calls, e.g. a thinking model that spent the whole max_output_tokens budget on
        # reasoning, so callers see the terminal reason instead of an empty choices list.
        if (
            tool_calls_list
            or text_content is not None
            or images
            or audio
            or code_execution
            or mapped_finish_reason in ("length", "content_filter")
        ):
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
    else:
        logger.debug(
            "generateContent response without candidates: response_id=%s model_version=%s prompt_feedback=%s usage=%s",
            response.response_id,
            response.model_version,
            response.prompt_feedback.model_dump_json(exclude_none=True) if response.prompt_feedback else None,
            response.usage_metadata.model_dump_json(exclude_none=True) if response.usage_metadata else None,
        )
        message = "generateContent returned no candidates and the prompt was not blocked"
        raise ProviderError(message, provider_name=provider_name)

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
    code_execution_state: CodeExecutionState | None = None,
) -> ChatCompletionChunk:
    """Convert a Google GenerateContentResponse to an OpenAI ChatCompletionChunk.

    Args:
        response: The Google GenerateContentResponse streaming chunk.
        tool_call_counter: Optional single-element list holding the number of tool
            calls already emitted earlier in the same stream. This should be created
            once per stream and passed in by the caller so that ``index`` stays stable
            and unique across chunks, rather than restarting at 0 for every chunk.
        code_execution_state: Optional per-stream state, like ``tool_call_counter``, so a
            code execution result in a later chunk pairs with the code emitted before it.
    """

    candidate = response.candidates[0] if response.candidates else None

    if tool_call_counter is None:
        tool_call_counter = [0]
    if code_execution_state is None:
        code_execution_state = CodeExecutionState()

    content = ""
    reasoning_content = ""
    tool_calls_list: list[ChoiceDeltaToolCall] = []
    message_extra_content = None
    code_execution: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    audio_blobs: list[types.Blob] = []

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
                    id=_tool_call_id(function_call, response.response_id, tool_call_index),
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
            if audio_blob := _inline_audio_blob(part):
                audio_blobs.append(audio_blob)
            if code_execution_item := _code_execution_item(part, code_execution_state):
                code_execution.append(code_execution_item)
            content += part.text or ""  # the signed final part may carry empty text
            message_extra_content = _thought_signature_extra_content(part) or message_extra_content

    message_extra_content = _merge_code_execution(message_extra_content, code_execution)

    audio = None
    if converted_audio := _inline_data_audio(audio_blobs, content, playable=False):
        audio = ChoiceDeltaAudio(data=converted_audio["data"], transcript=converted_audio["transcript"] or None)

    # Unmapped reasons stay None so non-final chunks are not forced to a terminal reason.
    mapped_finish_reason = _map_finish_reason(candidate.finish_reason) if candidate else None
    prompt_was_blocked = _prompt_was_blocked(response)
    # Gemini can send the function call and the STOP finish reason in separate chunks, so a
    # tool call emitted earlier in the stream must still turn the final "stop" into "tool_calls".
    # Only the terminal chunk reports it: a client that runs tools on each "tool_calls" would
    # otherwise run them twice.
    has_tool_calls = mapped_finish_reason is not None and tool_call_counter[0] > 0
    finish_reason = (
        "content_filter" if prompt_was_blocked else _resolve_finish_reason(mapped_finish_reason, has_tool_calls)
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

    usage = CompletionUsage(**_extract_usage_dict(response)) if response.usage_metadata else None

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
    model: str | None = None,
) -> types.InlinedRequest:
    """Convert a single OpenAI-format JSONL entry to a Google ``InlinedRequest``.

    The *entry* dict follows the OpenAI batch format with ``custom_id`` and
    ``body`` (containing ``model``, ``messages``, and optional parameters).
    A ``model`` given here overrides the body's model.
    """
    body = entry.get("body", {})
    messages = body.get("messages", [])
    model = model or body.get("model", "")

    contents, system_instruction = _convert_messages(messages, provider_name=provider_name, model=model)

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
