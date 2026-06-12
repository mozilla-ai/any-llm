from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, TypeGuard

from pydantic import BaseModel, TypeAdapter, ValidationError

from any_llm.logging import logger

if TYPE_CHECKING:
    from any_llm.types.responses import ParsedResponse, Response


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
        if isinstance(response, Response):
            # Some providers return a valid Response whose shape does not satisfy the stricter
            # ParsedResponse schema (e.g. Fireworks: usage.output_tokens_details=None). Build the
            # ParsedResponse without re-validating the whole payload so output_parsed still works.
            parsed = _parsed_response_from_raw(response)
        else:
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


def _parsed_response_from_raw(response: Response) -> ParsedResponse[Any]:
    """Build a ``ParsedResponse`` from an already-validated ``Response`` without re-validation.

    Used when a provider returns a valid ``Response`` that does not satisfy the stricter
    ``ParsedResponse`` schema. Message text content is rebuilt as ``ParsedResponseOutputText``
    (which carries ``parsed``) so the downstream parse loop and ``output_parsed`` keep working;
    other fields are carried over verbatim via ``model_construct`` (no coercion).
    """
    from openai.types.responses import (
        ParsedResponseOutputMessage,
        ParsedResponseOutputText,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    from any_llm.types.responses import ParsedResponse

    parsed_output: list[Any] = []
    for item in response.output:
        if not isinstance(item, ResponseOutputMessage):
            parsed_output.append(item)
            continue
        parsed_content: list[Any] = [
            ParsedResponseOutputText.model_construct(**content.model_dump(by_alias=True, warnings=False))
            if isinstance(content, ResponseOutputText)
            else content
            for content in item.content
        ]
        parsed_output.append(
            ParsedResponseOutputMessage.model_construct(
                **{**item.model_dump(by_alias=True, warnings=False), "content": parsed_content}
            )
        )

    data = response.model_dump(by_alias=True, warnings=False)
    data["output"] = parsed_output
    return ParsedResponse.model_construct(**data)
