import dataclasses
from typing import Annotated, Any, get_type_hints

import pytest
from pydantic import BaseModel, Field

from any_llm import tools
from any_llm.tools import callable_to_tool


@dataclasses.dataclass
class DataclassNode:
    name: str
    children: list["DataclassNode"] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DataclassLeft:
    name: str
    right: "DataclassRight | None" = None


@dataclasses.dataclass
class DataclassRight:
    count: int
    left: DataclassLeft | None = None


class ModelNode(BaseModel):
    name: str = Field(alias="label")
    children: list["ModelNode"] = Field(default_factory=list)


class ModelLeft(BaseModel):
    name: str
    right: "ModelRight | None" = None


class ModelRight(BaseModel):
    count: int
    left: ModelLeft | None = None


ModelLeft.model_rebuild()


@dataclasses.dataclass
class DataclassPair:
    first: DataclassNode
    second: DataclassNode


class ModelPair(BaseModel):
    first: ModelNode
    second: ModelNode


def _schema(model: Any) -> dict[str, Any]:
    def accept(item: model) -> None:
        """Accept a model."""

    result: dict[str, Any] = callable_to_tool(accept)["function"]["parameters"]["properties"]["item"]
    return result


@pytest.mark.parametrize(
    ("model", "name"),
    [(DataclassNode, "name"), (ModelNode, "label")],
)
def test_self_referencing_models_stop_at_the_cycle(model: type, name: str) -> None:
    schema = _schema(model)
    assert schema["properties"] == {
        name: {"type": "string"},
        "children": {"type": "array", "items": {"type": "object"}},
    }
    assert schema["required"] == [name]


@pytest.mark.parametrize("model", [DataclassLeft, ModelLeft])
def test_mutually_referencing_models_stop_at_the_cycle(model: type) -> None:
    schema = _schema(model)
    assert schema["properties"] == {
        "name": {"type": "string"},
        "right": {
            "type": "object",
            "properties": {"count": {"type": "integer"}, "left": {"type": "object"}},
            "required": ["count"],
        },
    }
    assert schema["required"] == ["name"]


@pytest.mark.parametrize("model", [DataclassPair, ModelPair])
def test_shared_model_fields_and_repeated_calls_expand_independently(model: type) -> None:
    first = _schema(model)
    assert first == _schema(model)
    assert first["properties"]["first"] == first["properties"]["second"]
    assert "properties" in first["properties"]["first"]
    assert first["required"] == ["first", "second"]


@pytest.mark.parametrize("model", [DataclassNode, ModelNode])
def test_annotated_recursive_models_use_the_same_cycle_guard(model: type) -> None:
    assert _schema(Annotated[model, "metadata"]) == _schema(model)


@pytest.mark.parametrize("model", [DataclassNode, ModelNode])
def test_model_expansion_state_is_reset_after_an_error(model: type, monkeypatch: pytest.MonkeyPatch) -> None:
    original = get_type_hints

    def fail_type_hints(obj: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if obj is model:
            message = "unresolvable model field"
            raise NameError(message)
        return original(obj, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(tools, "get_type_hints", fail_type_hints)
        with pytest.raises(NameError, match="unresolvable model field"):
            _schema(model)

    assert "properties" in _schema(model)
