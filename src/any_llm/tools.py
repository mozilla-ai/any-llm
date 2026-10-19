"""Tools utilities for converting Python callables to OpenAI tools format."""

from __future__ import annotations

import dataclasses
import enum
import inspect
import types as _types
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints
from typing import Annotated as _Annotated
from typing import Literal as _Literal
from typing import NotRequired as _NotRequired
from typing import Required as _Required

from pydantic import AliasChoices, AliasPath, ConfigDict
from pydantic import BaseModel as PydanticBaseModel
from typing_extensions import ReadOnly as _ReadOnly
from typing_extensions import is_typeddict as _is_typeddict

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

# A self-referencing TypedDict would otherwise expand forever once its forward references are resolved.
_EXPANDING_TYPED_DICTS: ContextVar[frozenset[Any]] = ContextVar("_EXPANDING_TYPED_DICTS", default=frozenset())

# Pydantic < 2.11 has no validate_by_alias and ignores the key, so aliases stay accepted there.
_PYDANTIC_HAS_VALIDATE_BY_ALIAS = "validate_by_alias" in ConfigDict.__annotations__


def callable_to_tool(func: Callable[..., Any]) -> dict[str, Any]:
    """Convert a Python callable to OpenAI tools format.

    Args:
        func: A Python callable (function) to convert to a tool

    Returns:
        Dictionary in OpenAI tools format

    Raises:
        ValueError: If the function doesn't have proper docstring or type annotations

    Example:
        >>> def get_weather(location: str, unit: str = "celsius") -> str:
        ...     '''Get weather information for a location.'''
        ...     return f"Weather in {location} is sunny, 25°{unit[0].upper()}"
        >>>
        >>> tool = callable_to_tool(get_weather)
        >>> # Returns OpenAI tools format dict

    """
    if not func.__doc__:
        msg = f"Function {func.__name__} must have a docstring"
        raise ValueError(msg)

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotated_type = type_hints.get(param_name, str)
        param_schema = _python_type_to_json_schema(annotated_type)

        type_name = getattr(annotated_type, "__name__", str(annotated_type))
        properties[param_name] = {
            **param_schema,
            "description": f"Parameter {param_name} of type {type_name}",
        }

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__.strip(),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _pydantic_input_name(name: str, field_info: FieldInfo, config: Mapping[str, Any]) -> str:
    """Return the key Pydantic accepts for a field on input.

    An AliasPath is only representable when it is a single string key; otherwise the field name is used.
    """
    if _PYDANTIC_HAS_VALIDATE_BY_ALIAS and not config.get("validate_by_alias", True):
        return name
    alias = field_info.validation_alias
    choices = alias.choices if isinstance(alias, AliasChoices) else [alias]
    for choice in choices:
        if isinstance(choice, str):
            return choice
        if isinstance(choice, AliasPath) and len(choice.path) == 1 and isinstance(choice.path[0], str):
            return choice.path[0]
    return name


def _python_type_to_json_schema(python_type: Any) -> dict[str, Any]:
    """Convert Python type annotation to a JSON Schema for a parameter.

    Supported mappings (subset tailored for LLM tool schemas):
    - Primitives: str/int/float/bool -> string/integer/number/boolean
    - bytes -> string with contentEncoding base64
    - datetime/date/time -> string with format date-time/date/time
    - list[T] / Sequence[T] / set[T] / frozenset[T] -> array with items=schema(T)
      - set/frozenset include uniqueItems=true
      - list without type args defaults items to string
    - dict[K,V] / Mapping[K,V] -> object with additionalProperties=schema(V)
      - dict without type args defaults additionalProperties to string
    - tuple[T1, T2, ...] -> array with prefixItems per element and min/maxItems
    - tuple[T, ...] -> array with items=schema(T)
    - Union[X, Y] and X | Y -> anyOf=[schema(X), schema(Y)] (without top-level type)
    - Optional[T] (Union[T, None]) -> schema(T) (nullability not encoded)
    - Literal[...]/Enum -> enum with appropriate type inference when uniform
    - TypedDict -> object with properties/required per annotations; Required/NotRequired/ReadOnly
      qualifiers are unwrapped, and a self-reference becomes a bare object
    - dataclass/Pydantic BaseModel -> object with nested properties inferred from fields;
      Pydantic fields are named the way Pydantic validates them (see _pydantic_input_name)
    """
    origin = get_origin(python_type)
    args = get_args(python_type)

    if _Annotated is not None and origin is _Annotated and len(args) >= 1:
        python_type = args[0]
        origin = get_origin(python_type)
        args = get_args(python_type)

    if python_type is Any:
        return {"type": "string"}

    primitive_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if python_type in primitive_map:
        return {"type": primitive_map[python_type]}

    if python_type is bytes:
        return {"type": "string", "contentEncoding": "base64"}
    if python_type is datetime:
        return {"type": "string", "format": "date-time"}
    if python_type is date:
        return {"type": "string", "format": "date"}
    if python_type is time:
        return {"type": "string", "format": "time"}

    if python_type is list:
        return {"type": "array", "items": {"type": "string"}}
    if python_type is dict:
        return {"type": "object", "additionalProperties": {"type": "string"}}

    if origin is _Literal:
        literal_values = list(args)
        schema_lit: dict[str, Any] = {"enum": literal_values}
        if all(isinstance(v, bool) for v in literal_values):
            schema_lit["type"] = "boolean"
        elif all(isinstance(v, str) for v in literal_values):
            schema_lit["type"] = "string"
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in literal_values):
            schema_lit["type"] = "integer"
        elif all(isinstance(v, int | float) and not isinstance(v, bool) for v in literal_values):
            schema_lit["type"] = "number"
        return schema_lit

    if inspect.isclass(python_type) and issubclass(python_type, enum.Enum):
        enum_values = [e.value for e in python_type]
        value_types = {type(v) for v in enum_values}
        schema: dict[str, Any] = {"enum": enum_values}
        if value_types == {str}:
            schema["type"] = "string"
        elif value_types == {int}:
            schema["type"] = "integer"
        elif value_types <= {int, float}:
            schema["type"] = "number"
        elif value_types == {bool}:
            schema["type"] = "boolean"
        return schema

    if _is_typeddict(python_type):
        expanding = _EXPANDING_TYPED_DICTS.get()
        if python_type in expanding:
            return {"type": "object"}
        annotations = _resolve_typed_dict_annotations(python_type)
        required_keys = set(getattr(python_type, "__required_keys__", set()))
        td_properties: dict[str, Any] = {}
        td_required: list[str] = []
        token = _EXPANDING_TYPED_DICTS.set(expanding | {python_type})
        try:
            for field_name, annotation in annotations.items():
                field_type, explicitly_required = _unwrap_typed_dict_field(annotation)
                td_properties[field_name] = _python_type_to_json_schema(field_type)
                # __required_keys__ is computed before postponed annotations are resolved, so an explicit
                # Required/NotRequired qualifier takes precedence over it.
                if explicitly_required is None:
                    explicitly_required = field_name in required_keys
                if explicitly_required:
                    td_required.append(field_name)
        finally:
            _EXPANDING_TYPED_DICTS.reset(token)
        schema_td: dict[str, Any] = {
            "type": "object",
            "properties": td_properties,
        }
        if td_required:
            schema_td["required"] = td_required
        return schema_td

    if inspect.isclass(python_type) and dataclasses.is_dataclass(python_type):
        type_hints = get_type_hints(python_type)
        # Pydantic dataclasses keep their FieldInfo (aliases, defaults) here, not on the stdlib field.
        dc_pydantic_fields: dict[str, FieldInfo] = getattr(python_type, "__pydantic_fields__", {})
        dc_config: Mapping[str, Any] = getattr(python_type, "__pydantic_config__", {})
        dc_properties: dict[str, Any] = {}
        dc_required: list[str] = []
        for field in dataclasses.fields(python_type):
            field_type = type_hints.get(field.name, Any)
            dc_field_info = dc_pydantic_fields.get(field.name)
            if dc_field_info is not None:
                dc_name = _pydantic_input_name(field.name, dc_field_info, dc_config)
                dc_is_required = dc_field_info.is_required()
            else:
                dc_name = field.name
                dc_is_required = (
                    field.default is dataclasses.MISSING
                    and getattr(field, "default_factory", dataclasses.MISSING) is dataclasses.MISSING
                )
            dc_properties[dc_name] = _python_type_to_json_schema(field_type)
            if dc_is_required:
                dc_required.append(dc_name)
        schema_dc: dict[str, Any] = {"type": "object", "properties": dc_properties}
        if dc_required:
            schema_dc["required"] = dc_required
        return schema_dc

    if inspect.isclass(python_type) and issubclass(python_type, PydanticBaseModel):
        model_type_hints = get_type_hints(python_type)
        pd_properties: dict[str, Any] = {}
        pd_required: list[str] = []
        model_fields = getattr(python_type, "model_fields", {})
        for name, field_info in model_fields.items():
            schema_name = _pydantic_input_name(name, field_info, python_type.model_config)
            pd_properties[schema_name] = _python_type_to_json_schema(model_type_hints.get(name, Any))
            is_required = getattr(field_info, "is_required", None)
            if callable(is_required) and is_required():
                pd_required.append(schema_name)
        schema_pd: dict[str, Any] = {"type": "object", "properties": pd_properties}
        if pd_required:
            schema_pd["required"] = pd_required
        return schema_pd

    if origin in (list, Sequence, set, frozenset):
        item_type = args[0] if args else Any
        item_schema = _python_type_to_json_schema(item_type)
        schema_arr: dict[str, Any] = {"type": "array", "items": item_schema or {"type": "string"}}
        if origin in (set, frozenset):
            schema_arr["uniqueItems"] = True
        return schema_arr
    if origin is tuple:
        if not args:
            return {"type": "array", "items": {"type": "string"}}
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        prefix_items = [_python_type_to_json_schema(a) for a in args]
        return {
            "type": "array",
            "prefixItems": prefix_items,
            "minItems": len(prefix_items),
            "maxItems": len(prefix_items),
        }

    if origin in (dict, Mapping):
        value_type = args[1] if len(args) >= 2 else Any
        value_schema = _python_type_to_json_schema(value_type)
        return {"type": "object", "additionalProperties": value_schema or {"type": "string"}}

    typing_union = getattr(__import__("typing"), "Union", None)
    if origin in (typing_union, _types.UnionType):
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) > 1:
            schemas = [_python_type_to_json_schema(arg) for arg in non_none_args]
            # Not oneOf: branches can overlap (int and float, str and datetime), and oneOf
            # rejects a value that matches more than one branch.
            return {"anyOf": schemas}
        if non_none_args:
            return _python_type_to_json_schema(non_none_args[0])
        return {"type": "string"}

    return {"type": "string"}


def _resolve_typed_dict_annotations(typed_dict: Any) -> dict[str, Any]:
    """Resolve a TypedDict's field annotations, keeping the raw annotation for any field that can't be resolved."""
    try:
        return get_type_hints(typed_dict, include_extras=True)
    except NameError:
        pass
    # Names imported only under TYPE_CHECKING make get_type_hints fail for the whole class, so resolve
    # each field on its own through a stand-in class that evaluates in the TypedDict's module.
    annotations: dict[str, Any] = {}
    for field_name, annotation in (getattr(typed_dict, "__annotations__", {}) or {}).items():
        holder = type(
            typed_dict.__name__, (), {"__annotations__": {field_name: annotation}, "__module__": typed_dict.__module__}
        )
        try:
            annotations[field_name] = get_type_hints(holder, include_extras=True)[field_name]
        except NameError:
            annotations[field_name] = annotation
    return annotations


def _unwrap_typed_dict_field(annotation: Any) -> tuple[Any, bool | None]:
    """Strip Annotated/ReadOnly/Required/NotRequired from a TypedDict field.

    Returns the underlying type and True/False when Required/NotRequired is present, None otherwise.
    """
    explicitly_required: bool | None = None
    while True:
        origin = get_origin(annotation)
        if origin in (_Annotated, _ReadOnly, _Required, _NotRequired):
            if origin is _Required:
                explicitly_required = True
            elif origin is _NotRequired:
                explicitly_required = False
            annotation = get_args(annotation)[0]
        else:
            return annotation, explicitly_required


def _flatten_responses_tool(tool: Any) -> Any:
    """Flatten a chat-format function tool for the Responses API, which takes function tools flat."""
    if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return {"type": "function", **tool["function"]}
    return tool


def prepare_tools(
    tools: list[dict[str, Any] | Callable[..., Any] | Any], built_in_tools: list[Any] | None = None
) -> list[dict[str, Any] | Any]:
    """Prepare tools for completion API by converting callables to OpenAI format.

    Args:
        tools: List of tools, can be mix of callables and already formatted tool dicts
        built_in_tools: Optional list of built-in tool instances to include as-is
            For example, in `gemini` provider, you can pass `types.Tool(google_search=types.GoogleSearch())`.

    Returns:
        List of tools in OpenAI format

    Example:
        >>> def add(a: int, b: int) -> int:
        ...     '''Add two numbers.'''
        ...     return a + b
        >>>
        >>> tools = prepare_tools([add])
        >>> # Returns list of OpenAI format tool dicts

    """
    prepared_tools = []

    for tool in tools:
        if built_in_tools and any(isinstance(tool, b) for b in built_in_tools):
            prepared_tools.append(tool)
        elif callable(tool):
            prepared_tools.append(callable_to_tool(tool))
        elif isinstance(tool, dict):
            prepared_tools.append(tool)
        else:
            msg = f"Tool must be callable or dict, got {type(tool)}"
            raise ValueError(msg)

    return prepared_tools
