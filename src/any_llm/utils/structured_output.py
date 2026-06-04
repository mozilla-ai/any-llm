from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, TypeGuard

from pydantic import BaseModel, TypeAdapter, ValidationError

from any_llm.logging import logger

if TYPE_CHECKING:
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


def build_responses_text_format(response_format: type) -> dict[str, Any]:
    """Build the Responses API ``text`` config that requests schema-conformant JSON.

    Used by providers that cannot call ``client.responses.parse()`` (e.g. the
    OpenResponses providers) so the model still emits JSON matching the schema.
    Works with both Pydantic BaseModel subclasses and dataclass types.
    """
    return {
        "format": {
            "type": "json_schema",
            "name": response_format.__name__,
            "schema": get_json_schema(response_format),
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
        parsed: ParsedResponse[Any] = ParsedResponse.model_validate(response.model_dump(warnings=False))
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
