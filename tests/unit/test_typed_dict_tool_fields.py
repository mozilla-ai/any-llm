from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, NotRequired, Required, TypedDict

from typing_extensions import ReadOnly
from typing_extensions import TypedDict as ExtTypedDict

from any_llm.tools import callable_to_tool

if TYPE_CHECKING:
    from decimal import Decimal


class Options(TypedDict, total=False):
    count: Required[int]
    threshold: NotRequired[float]
    enabled: bool


def configure(options: Options) -> None:
    """Configure a task."""


def test_typed_dict_qualifiers_keep_underlying_field_types() -> None:
    schema = callable_to_tool(configure)["function"]["parameters"]["properties"]["options"]
    assert schema["properties"] == {
        "count": {"type": "integer"},
        "threshold": {"type": "number"},
        "enabled": {"type": "boolean"},
    }
    assert schema["required"] == ["count"]


class AnnotatedOptions(TypedDict):
    count: Annotated[Required[int], "count metadata"]
    label: Annotated[NotRequired[str], "label metadata"]


def configure_annotated(options: AnnotatedOptions) -> None:
    """Configure a task with annotated fields."""


def test_annotated_typed_dict_qualifiers_override_postponed_metadata() -> None:
    schema = callable_to_tool(configure_annotated)["function"]["parameters"]["properties"]["options"]
    assert schema["properties"] == {"count": {"type": "integer"}, "label": {"type": "string"}}
    assert schema["required"] == ["count"]


class ReadOnlyOptions(ExtTypedDict, total=False):
    count: ReadOnly[Required[int]]
    ratio: Required[ReadOnly[float]]
    label: ReadOnly[str]


def configure_read_only(options: ReadOnlyOptions) -> None:
    """Configure a task with read-only fields."""


def test_read_only_typed_dict_fields_keep_types_and_required_qualifiers() -> None:
    schema = callable_to_tool(configure_read_only)["function"]["parameters"]["properties"]["options"]
    assert schema["properties"] == {
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "label": {"type": "string"},
    }
    assert schema["required"] == ["count", "ratio"]


class Node(TypedDict):
    name: str
    children: list[Node]


def walk(tree: Node) -> None:
    """Walk a tree."""


def test_self_referencing_typed_dict_stops_at_the_cycle() -> None:
    schema = callable_to_tool(walk)["function"]["parameters"]["properties"]["tree"]
    assert schema["properties"] == {
        "name": {"type": "string"},
        "children": {"type": "array", "items": {"type": "object"}},
    }
    assert schema["required"] == ["name", "children"]


class Payment(TypedDict):
    amount: Decimal
    currency: str


def pay(payment: Payment) -> None:
    """Make a payment."""


def test_typed_dict_with_unresolvable_field_falls_back_to_raw_annotations() -> None:
    schema = callable_to_tool(pay)["function"]["parameters"]["properties"]["payment"]
    assert schema["properties"] == {"amount": {"type": "string"}, "currency": {"type": "string"}}
    assert schema["required"] == ["amount", "currency"]
