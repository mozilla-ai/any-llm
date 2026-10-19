from typing import Any

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass as pydantic_dataclass

from any_llm.tools import callable_to_tool


class Aliased(BaseModel):
    value: int = Field(alias="public_value")


class ValidationAliased(BaseModel):
    value: int = Field(validation_alias="input_value")


class BothAliases(BaseModel):
    value: int = Field(alias="output_value", validation_alias="input_value")


class ChoicesAliased(BaseModel):
    value: int = Field(alias="output_value", validation_alias=AliasChoices("first", "second"))


class ChoicesStartingWithPath(BaseModel):
    value: int = Field(validation_alias=AliasChoices(AliasPath("nested", 0), "flat"))


class SingleKeyPath(BaseModel):
    value: int = Field(validation_alias=AliasPath("key"))


class NestedPath(BaseModel):
    model_config = ConfigDict(validate_by_name=True)

    value: int = Field(validation_alias=AliasPath("nested", 0))


class ValidateByNameOnly(BaseModel):
    model_config = ConfigDict(validate_by_alias=False, validate_by_name=True)

    value: int = Field(alias="public_value")


class GeneratedAliases(BaseModel):
    model_config = ConfigDict(alias_generator=lambda name: name.upper())

    value: int


class OptionalAliased(BaseModel):
    value: int = Field(default=0, alias="public_value")


@pydantic_dataclass
class DataclassAliased:
    value: int = Field(alias="public_value")


@pydantic_dataclass
class DataclassOptionalAliased:
    value: int = Field(default=0, alias="public_value")


def _item_schema(model: type) -> dict[str, Any]:
    def tool(item: model) -> None:  # type: ignore[valid-type]
        """Accept a model."""

    schema: dict[str, Any] = callable_to_tool(tool)["function"]["parameters"]["properties"]["item"]
    return schema


@pytest.mark.parametrize(
    ("model", "expected_name"),
    [
        (Aliased, "public_value"),
        (ValidationAliased, "input_value"),
        (BothAliases, "input_value"),
        (ChoicesAliased, "first"),
        (ChoicesStartingWithPath, "flat"),
        (SingleKeyPath, "key"),
        (NestedPath, "value"),
        (ValidateByNameOnly, "value"),
        (GeneratedAliases, "VALUE"),
        (DataclassAliased, "public_value"),
    ],
)
def test_tool_model_fields_use_pydantic_input_names(model: type, expected_name: str) -> None:
    schema = _item_schema(model)
    assert schema["properties"] == {expected_name: {"type": "integer"}}
    assert schema["required"] == [expected_name]
    assert TypeAdapter(model).validate_python({expected_name: 3}).value == 3


@pytest.mark.parametrize("model", [OptionalAliased, DataclassOptionalAliased])
def test_tool_model_fields_with_defaults_are_not_required(model: type) -> None:
    schema = _item_schema(model)
    assert schema["properties"] == {"public_value": {"type": "integer"}}
    assert "required" not in schema
