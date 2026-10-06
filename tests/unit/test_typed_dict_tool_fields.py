from __future__ import annotations

from typing import Annotated, NotRequired, Required, TypedDict

from any_llm.tools import callable_to_tool


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
