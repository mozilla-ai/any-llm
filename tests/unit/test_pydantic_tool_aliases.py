from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel, Field

from any_llm.tools import callable_to_tool


class Aliased(BaseModel):
    value: int = Field(alias="public_value")


class ValidationAliased(BaseModel):
    value: int = Field(validation_alias="input_value")


class BothAliases(BaseModel):
    value: int = Field(alias="output_value", validation_alias="input_value")


def alias_tool(item: Aliased) -> None:
    """Accept an aliased model."""


def validation_alias_tool(item: ValidationAliased) -> None:
    """Accept a model with a validation alias."""


def both_aliases_tool(item: BothAliases) -> None:
    """Accept a model with distinct input and output aliases."""


@pytest.mark.parametrize(
    ("function", "model"),
    [(alias_tool, Aliased), (validation_alias_tool, ValidationAliased), (both_aliases_tool, BothAliases)],
)
def test_tool_model_fields_use_pydantic_input_names(function: Callable[..., Any], model: type[BaseModel]) -> None:
    schema = callable_to_tool(function)["function"]["parameters"]["properties"]["item"]
    reference = model.model_json_schema()
    assert set(schema["properties"]) == set(reference["properties"])
    assert schema["required"] == reference["required"]
    field_name = next(iter(reference["properties"]))
    assert schema["properties"][field_name]["type"] == "integer"
    assert model.model_validate({field_name: 3}).model_dump() == {"value": 3}
