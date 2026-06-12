import dataclasses
from typing import Any

from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from pydantic import BaseModel

from any_llm.types.responses import ParsedResponse, Response
from any_llm.utils.structured_output import (
    build_responses_text_format,
    get_json_schema,
    is_structured_output_type,
    parse_json_content,
    parse_responses_output,
)


def _make_response(text: str) -> Response:
    message = ResponseOutputMessage(
        id="msg-1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )
    return Response(
        id="resp-1",
        created_at=0,
        model="test-model",
        object="response",
        output=[message],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


class PydanticModel(BaseModel):
    name: str
    age: int


@dataclasses.dataclass
class DataclassModel:
    name: str
    age: int


def test_is_structured_output_type_basemodel() -> None:
    assert is_structured_output_type(PydanticModel) is True


def test_is_structured_output_type_dataclass() -> None:
    assert is_structured_output_type(DataclassModel) is True


def test_is_structured_output_type_dict() -> None:
    assert is_structured_output_type({"type": "json_object"}) is False


def test_is_structured_output_type_none() -> None:
    assert is_structured_output_type(None) is False


def test_is_structured_output_type_str() -> None:
    assert is_structured_output_type(str) is False


def test_is_structured_output_type_instance() -> None:
    assert is_structured_output_type(DataclassModel(name="test", age=1)) is False


def test_get_json_schema_basemodel() -> None:
    schema = get_json_schema(PydanticModel)
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]


def test_get_json_schema_dataclass() -> None:
    schema = get_json_schema(DataclassModel)
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]


def test_parse_json_content_basemodel() -> None:
    result = parse_json_content(PydanticModel, '{"name": "Alice", "age": 30}')
    assert isinstance(result, PydanticModel)
    assert result.name == "Alice"
    assert result.age == 30


def test_parse_json_content_dataclass() -> None:
    result = parse_json_content(DataclassModel, '{"name": "Alice", "age": 30}')
    assert isinstance(result, DataclassModel)
    assert result.name == "Alice"
    assert result.age == 30


def test_parse_json_content_dataclass_nested() -> None:
    @dataclasses.dataclass
    class Address:
        city: str
        country: str

    @dataclasses.dataclass
    class Person:
        name: str
        address: Address

    result: Any = parse_json_content(Person, '{"name": "Alice", "address": {"city": "Paris", "country": "France"}}')
    assert isinstance(result, Person)
    assert result.name == "Alice"
    assert isinstance(result.address, Address)
    assert result.address.city == "Paris"


def test_build_responses_text_format_basemodel() -> None:
    text_config = build_responses_text_format(PydanticModel)
    fmt = text_config["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["name"] == "PydanticModel"
    assert "name" in fmt["schema"]["properties"]


def test_build_responses_text_format_dataclass() -> None:
    text_config = build_responses_text_format(DataclassModel)
    assert text_config["format"]["name"] == "DataclassModel"
    assert "age" in text_config["format"]["schema"]["properties"]


def test_build_responses_text_format_is_strict() -> None:
    """The text format must be strict so OpenAI accepts it and lenient providers echo the schema back."""
    for model in (PydanticModel, DataclassModel):
        fmt = build_responses_text_format(model)["format"]
        assert fmt["strict"] is True
        assert fmt["schema"]["additionalProperties"] is False


def test_parse_responses_output_roundtrips_json_schema_text_format() -> None:
    """Regression: a json_schema text.format must survive the model_dump round-trip.

    The OpenAI type stores the schema under the Python attribute ``schema_`` with an alias of
    ``schema``; dumping without ``by_alias=True`` drops it and re-validation fails.
    """
    from openai.types.responses import ResponseFormatTextJSONSchemaConfig
    from openai.types.responses.response_text_config import ResponseTextConfig

    response = _make_response('{"name": "Carol", "age": 5}')
    response.text = ResponseTextConfig(
        format=ResponseFormatTextJSONSchemaConfig.model_validate(
            {
                "name": "DataclassModel",
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                "type": "json_schema",
                "strict": True,
            }
        )
    )

    parsed = parse_responses_output(response, DataclassModel)
    assert isinstance(parsed, ParsedResponse)
    assert parsed.output_parsed.age == 5


def test_parse_responses_output_basemodel() -> None:
    parsed = parse_responses_output(_make_response('{"name": "Alice", "age": 30}'), PydanticModel)
    assert isinstance(parsed, ParsedResponse)
    assert isinstance(parsed.output_parsed, PydanticModel)
    assert parsed.output_parsed.name == "Alice"


def test_parse_responses_output_dataclass() -> None:
    parsed = parse_responses_output(_make_response('{"name": "Bob", "age": 7}'), DataclassModel)
    assert isinstance(parsed, ParsedResponse)
    assert isinstance(parsed.output_parsed, DataclassModel)
    assert parsed.output_parsed.age == 7


def test_parse_responses_output_returns_none_when_not_normalizable() -> None:
    class NotAResponse(BaseModel):
        output: list[Any] = []

    result = parse_responses_output(NotAResponse(), PydanticModel)
    assert result is None


def test_parse_responses_output_skips_non_message_output() -> None:
    """Non-message outputs and non-text content (e.g. tool calls, refusals) are skipped; the message is parsed."""
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputRefusal

    function_call = ResponseFunctionToolCall(
        id="fc-1", type="function_call", call_id="call-1", name="do_thing", arguments="{}", status="completed"
    )
    message = ResponseOutputMessage(
        id="msg-1",
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputRefusal(type="refusal", refusal="n/a"),
            ResponseOutputText(type="output_text", text='{"name": "Alice", "age": 30}', annotations=[]),
        ],
    )
    response = Response(
        id="resp-1",
        created_at=0,
        model="test-model",
        object="response",
        output=[function_call, message],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )

    parsed = parse_responses_output(response, PydanticModel)
    assert isinstance(parsed, ParsedResponse)
    assert isinstance(parsed.output_parsed, PydanticModel)
    assert parsed.output_parsed.name == "Alice"
