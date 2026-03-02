import dataclasses
from typing import Any

from pydantic import BaseModel

from any_llm.utils.structured_output import get_json_schema, is_structured_output_type, parse_json_content


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
