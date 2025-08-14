import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Annotated, Any, Literal, NotRequired, TypedDict

import pytest
from pydantic import BaseModel

from any_llm.tools import callable_to_tool, prepare_tools


def test_callable_to_tool_basic() -> None:
    """Test basic callable to tool conversion."""

    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    tool = callable_to_tool(add_numbers)

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "add_numbers"
    assert tool["function"]["description"] == "Add two numbers together."

    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"a", "b"}
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "integer"


def test_callable_to_tool_with_optional_params() -> None:
    """Test callable with optional parameters."""

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    tool = callable_to_tool(greet)

    params = tool["function"]["parameters"]
    assert params["required"] == ["name"]  # Only required parameter
    assert "greeting" in params["properties"]
    assert params["properties"]["name"]["type"] == "string"
    assert params["properties"]["greeting"]["type"] == "string"


def test_callable_to_tool_missing_docstring() -> None:
    """Test that function without docstring raises error."""

    def no_doc_function(x: int) -> int:
        return x

    with pytest.raises(ValueError, match="must have a docstring"):
        callable_to_tool(no_doc_function)


def test_prepare_tools_mixed() -> None:
    """Test prepare_tools with mix of callables and dicts."""

    def my_function(x: int) -> int:
        """My function."""
        return x

    existing_tool = {
        "type": "function",
        "function": {
            "name": "existing_tool",
            "description": "An existing tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    tools = prepare_tools([my_function, existing_tool])

    assert len(tools) == 2
    assert tools[0]["function"]["name"] == "my_function"
    assert tools[1]["function"]["name"] == "existing_tool"


def test_callable_to_tool() -> None:
    """Ensure list/dict annotations produce items/additionalProperties."""

    class AnArg(BaseModel):
        thing: str

    class UserTD(TypedDict):
        id: int
        name: NotRequired[str]

    @dataclass
    class UserDC:
        a: int
        b: str = "x"

    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    def another_tool(
        country: str,
        listing: list,  # type: ignore[type-arg]
        dicting: dict,  # type: ignore[type-arg]
        pydantic_arg: AnArg,
        list_specified: list[float],
        dict_specified: dict[str, int],
        union_specified: str | int,
        maybe_text: str | None = None,
        maybe_anything: Any = None,
        text: str = "t",
        number: int = 1,
        real: float = 1.0,
        flag: bool = False,
        data: bytes = b"x",
        ts: datetime = datetime(2000, 1, 1),  # noqa: DTZ001
        d: date = date(2000, 1, 1),
        t: time = time(0, 0, 0),
        set_nums: set[int] = frozenset({1}),  # type: ignore[assignment]
        frozenset_text: frozenset[str] = frozenset({"a"}),
        seq_nums: Sequence[int] = (),
        tup_any: tuple = tuple(),  # type: ignore[type-arg]  # noqa: C408
        tup_var: tuple[int, ...] = (1, 2, 3),
        tup_fixed: tuple[int, str] = (1, "a"),
        mapping_generic: Mapping[str, bool] = {},
        typed: UserTD = {"id": 1},  # noqa: B006
        data_class: UserDC = UserDC(1),  # noqa: B008
        literal_meta: Annotated[str, "meta"] | None = None,
        literal_ab: Literal["a", "b"] = "a",
        enum_color: Color = Color.RED,
        annotated_int: Annotated[int, "meta"] = 1,
        opt_int: int | None = None,
        union_three: str | float | bool = 1,
    ) -> None:
        """This is a docstring"""
        return

    tool = callable_to_tool(another_tool)

    params = tool["function"]["parameters"]
    props = params["properties"]

    assert props["listing"]["type"] == "array"
    assert "items" in props["listing"]

    assert props["dicting"]["type"] == "object"
    assert "additionalProperties" in props["dicting"]
    assert props["dicting"]["additionalProperties"]["type"] == "string"

    assert props["list_specified"]["type"] == "array"
    assert props["list_specified"]["items"]["type"] == "number"

    assert props["dict_specified"]["type"] == "object"
    assert props["dict_specified"]["additionalProperties"]["type"] == "integer"

    assert "oneOf" in props["union_specified"]
    assert len(props["union_specified"]["oneOf"]) == 2
    assert props["union_specified"]["oneOf"][0]["type"] == "string"
    assert props["union_specified"]["oneOf"][1]["type"] == "integer"

    assert props["maybe_text"]["type"] == "string"
    assert "maybe_text" not in params["required"]

    # Any type defaults to string and is not required when default provided
    assert props["maybe_anything"]["type"] == "string"
    assert "maybe_anything" not in params["required"]

    assert props["pydantic_arg"]["type"] == "object"
    assert props["pydantic_arg"]["properties"]["thing"]["type"] == "string"

    assert props["text"]["type"] == "string"
    assert props["number"]["type"] == "integer"
    assert props["real"]["type"] == "number"
    assert props["flag"]["type"] == "boolean"

    assert props["data"]["type"] == "string"
    assert props["data"]["contentEncoding"] == "base64"

    assert props["ts"]["type"] == "string"
    assert props["ts"]["format"] == "date-time"
    assert props["d"]["type"] == "string"
    assert props["d"]["format"] == "date"
    assert props["t"]["type"] == "string"
    assert props["t"]["format"] == "time"

    assert props["set_nums"]["type"] == "array"
    assert props["set_nums"]["items"]["type"] == "integer"
    assert props["set_nums"]["uniqueItems"] is True

    assert props["frozenset_text"]["type"] == "array"
    assert props["frozenset_text"]["items"]["type"] == "string"
    assert props["frozenset_text"]["uniqueItems"] is True

    assert props["seq_nums"]["type"] == "array"
    assert props["seq_nums"]["items"]["type"] == "integer"

    assert props["tup_any"]["type"] == "string"

    assert props["tup_var"]["type"] == "array"
    assert props["tup_var"]["items"]["type"] == "integer"

    assert props["tup_fixed"]["type"] == "array"
    assert len(props["tup_fixed"]["prefixItems"]) == 2
    assert props["tup_fixed"]["prefixItems"][0]["type"] == "integer"
    assert props["tup_fixed"]["prefixItems"][1]["type"] == "string"
    assert props["tup_fixed"]["minItems"] == 2
    assert props["tup_fixed"]["maxItems"] == 2

    assert props["mapping_generic"]["type"] == "object"
    assert props["mapping_generic"]["additionalProperties"]["type"] == "boolean"

    assert props["typed"]["type"] == "object"
    assert props["typed"]["properties"]["id"]["type"] == "integer"
    assert "required" in props["typed"]
    assert "id" in props["typed"]["required"]

    assert props["data_class"]["type"] == "object"
    assert props["data_class"]["properties"]["a"]["type"] == "integer"
    assert props["data_class"]["properties"]["b"]["type"] == "string"
    assert "a" in props["data_class"].get("required", [])

    assert props["literal_meta"]["type"] == "string"

    assert props["literal_ab"]["enum"] == ["a", "b"]
    assert props["literal_ab"]["type"] == "string"

    assert props["enum_color"]["enum"] == ["red", "blue"]
    assert props["enum_color"]["type"] == "string"

    assert props["annotated_int"]["type"] == "integer"

    assert props["opt_int"]["type"] == "integer"

    assert "oneOf" in props["union_three"]
    assert len(props["union_three"]["oneOf"]) == 3
    assert {s["type"] for s in props["union_three"]["oneOf"]} == {"string", "number", "boolean"}
