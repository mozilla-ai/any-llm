from collections.abc import Callable
from typing import Any

import pytest
from pydantic import TypeAdapter

from any_llm.tools import callable_to_tool


def number(value: float) -> None:
    """Accept a numeric value."""


def numbers(value: list[int | float]) -> None:
    """Accept a list of numeric values."""


def named_numbers(value: dict[str, int | float]) -> None:
    """Accept named numeric values."""


@pytest.mark.parametrize(
    ("function", "annotation"),
    [(number, float), (numbers, list[float]), (named_numbers, dict[str, float])],
)
def test_tool_numeric_unions_match_reference_schema(function: Callable[..., Any], annotation: Any) -> None:
    parameter = callable_to_tool(function)["function"]["parameters"]["properties"]["value"]
    schema = {key: value for key, value in parameter.items() if key != "description"}
    assert schema == TypeAdapter(annotation).json_schema()
