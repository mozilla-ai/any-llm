from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, TypeGuard

from pydantic import BaseModel, TypeAdapter, ValidationError

from any_llm.logging import logger

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types.parsed_message import ParsedMessage

    from any_llm.types.responses import ParsedResponse


def is_structured_output_type(response_format: Any) -> TypeGuard[type]:
    """Check if a type can be used for structured output parsing.

    Returns True for Pydantic BaseModel subclasses and dataclass types.
    """
    if not isinstance(response_format, type):
        return False
    if issubclass(response_format, BaseModel):
        return True
    if dataclasses.is_dataclass(response_format):
        return True
    return False


def get_json_schema(response_format: type) -> dict[str, Any]:
    """Get JSON schema from a structured output type.

    Works with both Pydantic BaseModel subclasses and dataclass types.
    """
    if issubclass(response_format, BaseModel):
        return response_format.model_json_schema()
    return TypeAdapter(response_format).json_schema()


def parse_json_content(response_format: type, content: str) -> Any:
    """Parse JSON content into the given structured type.

    Works with both Pydantic BaseModel subclasses and dataclass types.
    """
    if issubclass(response_format, BaseModel):
        return response_format.model_validate_json(content)
    return TypeAdapter(response_format).validate_json(content)


def _make_schema_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively enforce ``additionalProperties: false`` and require every property.

    Fallback used only if OpenAI's private strict-schema helper is unavailable; covers the
    common object/array/$defs cases (it does not rewrite optional fields as nullable).
    """

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("type") == "object" and "properties" in obj:
                obj["additionalProperties"] = False
                obj["required"] = list(obj["properties"].keys())
            for value in obj.values():
                _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(schema)
    return schema


def get_strict_json_schema(response_format: type) -> dict[str, Any]:
    """Get a strict JSON schema (``additionalProperties: false``, all fields required).

    Mirrors the schema the OpenAI SDK emits for ``responses.parse()``/``chat.completions.parse()``
    so the manual dataclass path stays byte-for-byte compatible with the native BaseModel path.
    Works with both Pydantic BaseModel subclasses and dataclass types.
    """
    schema = get_json_schema(response_format)
    try:
        from openai.lib._pydantic import _ensure_strict_json_schema
    except ImportError:  # pragma: no cover - private SDK helper relocated/removed
        return _make_schema_strict(schema)
    return _ensure_strict_json_schema(schema, path=(), root=schema)


def build_responses_text_format(response_format: type) -> dict[str, Any]:
    """Build the Responses API ``text`` config that requests schema-conformant JSON.

    Used by providers that cannot call ``client.responses.parse()`` (e.g. the
    OpenResponses providers) so the model still emits JSON matching the schema.
    The schema is made strict to match what ``responses.parse()`` sends for
    BaseModel types: OpenAI requires ``additionalProperties: false``, and lenient
    providers (Groq, HuggingFace) only echo back a complete ``text.format`` (with
    the ``schema`` field) when ``strict`` is set. Works with both Pydantic
    BaseModel subclasses and dataclass types.
    """
    return {
        "format": {
            "type": "json_schema",
            "strict": True,
            "name": response_format.__name__,
            "schema": get_strict_json_schema(response_format),
        }
    }


def parse_responses_output(response: Any, response_format: type) -> ParsedResponse[Any] | None:
    """Normalize a raw Responses result into a ``ParsedResponse`` with ``output_parsed``.

    Takes an OpenAI ``Response`` (what every provider yields on the structured-output
    path) and returns an OpenAI ``ParsedResponse`` whose ``output_parsed`` holds the
    typed object. Works with both Pydantic BaseModel subclasses and dataclass types.

    Returns ``None`` (after logging a warning) if ``response`` cannot be normalized into
    a ``ParsedResponse`` (e.g. an OpenResponses ``ResponseResource`` whose schema diverges
    from OpenAI's), so callers can fall back to the original object.
    """
    from openai.types.responses import ParsedResponseOutputMessage, ParsedResponseOutputText

    from any_llm.types.responses import ParsedResponse, Response

    try:
        # by_alias=True preserves alias-only fields (e.g. ResponseFormatTextJSONSchemaConfig.schema,
        # whose Python attribute is schema_) so the round-trip re-validates cleanly.
        parsed: ParsedResponse[Any] = ParsedResponse.model_validate(response.model_dump(by_alias=True, warnings=False))
    except ValidationError:
        if isinstance(response, Response):  # pragma: no cover - a valid Response always validates as ParsedResponse
            raise
        logger.warning(
            "Could not normalize %s into a ParsedResponse; returning it without output_parsed.",
            type(response).__name__,
        )
        return None

    for output in parsed.output:
        if not isinstance(output, ParsedResponseOutputMessage):
            continue
        for content in output.content:
            if isinstance(content, ParsedResponseOutputText) and content.text and content.parsed is None:
                content.parsed = parse_json_content(response_format, content.text)
    return parsed


def build_parsed_message(message: AnthropicMessage, output_format: type) -> ParsedMessage[Any]:
    """Build Anthropic's ``ParsedMessage`` from a Messages API response.

    Mirrors what ``client.messages.parse`` returns: each text block becomes a
    ``ParsedTextBlock`` whose ``parsed_output`` holds the typed object, and the message's
    ``parsed_output`` property surfaces the first one. Used for providers without a native
    parse helper so the structured-output result shape matches Anthropic's exactly.
    Works with both Pydantic BaseModel subclasses and dataclass types.
    """
    from anthropic.types import TextBlock
    from anthropic.types.parsed_message import ParsedMessage, ParsedTextBlock

    # Parametrize the generics at runtime so pydantic validates parsed_output against the
    # requested type; route through Any since a type held in a variable is not valid as a
    # static type parameter.
    parsed_text_block: Any = ParsedTextBlock
    parsed_message: Any = ParsedMessage
    text_block_cls = parsed_text_block[output_format]
    message_cls = parsed_message[output_format]

    content: list[Any] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parsed = parse_json_content(output_format, block.text) if block.text else None
            content.append(
                text_block_cls(type="text", text=block.text, citations=block.citations, parsed_output=parsed)
            )
        else:
            content.append(block)

    data = message.model_dump(exclude={"content"})
    result: ParsedMessage[Any] = message_cls(**data, content=content)
    return result
