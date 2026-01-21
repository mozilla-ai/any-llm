# Tools

This document describes how to convert native functions to OpenAI tool format for LLM function calling.

A helper `function_to_tool` must be implemented and exposed for the users.

## Overview

Function calling allows LLMs to request execution of user-defined functions. The model doesn't execute functions directly—it generates structured JSON that your application uses to call the appropriate function.

## OpenAI Tool Format

A tool definition has this structure:

```json
{
    "type": "function",
    "function": {
        "name": "function_name",
        "description": "Clear description of what the function does",
        "parameters": {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "Description of the parameter"
                }
            },
            "required": ["param_name"]
        }
    }
}
```

### Function Properties

| Property | Description |
|----------|-------------|
| `type` | Always `"function"` |
| `function.name` | The function's name (e.g., `get_weather`) |
| `function.description` | Details on when and how to use the function |
| `function.parameters` | JSON Schema defining the function's input arguments |

## Converting Native Functions

### Required Information

To convert a native function to OpenAI tool format, extract:

1. **Function name**: Use the function's identifier
2. **Description**: Extract from docstring/documentation comments
3. **Parameters**: Parse function signature for parameter names, types, and defaults
4. **Required parameters**: Parameters without default values

### Algorithm

```
function function_to_tool(func):
    1. Validate function has documentation (raise error if missing)
    2. Get function signature and type hints
    3. For each parameter (excluding *args, **kwargs):
        a. Get the annotated type (default to string if missing)
        b. Convert type to JSON Schema using type_to_json_schema()
        c. Add parameter description
        d. If no default value, mark as required
    4. Return tool definition object
```

## Type to JSON Schema Mapping

Convert language-native types to JSON Schema types.

### Primitive Types

| Native Type | JSON Schema |
|-------------|-------------|
| `string` / `str` | `{"type": "string"}` |
| `int` / `integer` | `{"type": "integer"}` |
| `float` / `number` | `{"type": "number"}` |
| `bool` / `boolean` | `{"type": "boolean"}` |

### Special String Types

| Native Type | JSON Schema |
|-------------|-------------|
| `bytes` | `{"type": "string", "contentEncoding": "base64"}` |
| `datetime` | `{"type": "string", "format": "date-time"}` |
| `date` | `{"type": "string", "format": "date"}` |
| `time` | `{"type": "string", "format": "time"}` |

### Collection Types

#### Arrays (List, Sequence, Set)

```json
// list[T] or Sequence[T]
{"type": "array", "items": <schema(T)>}

// set[T] or frozenset[T]
{"type": "array", "items": <schema(T)>, "uniqueItems": true}

// list without type args
{"type": "array", "items": {"type": "string"}}
```

#### Tuples

```json
// tuple[T1, T2, T3] - fixed length
{
    "type": "array",
    "prefixItems": [<schema(T1)>, <schema(T2)>, <schema(T3)>],
    "minItems": 3,
    "maxItems": 3
}

// tuple[T, ...] - variable length homogeneous
{"type": "array", "items": <schema(T)>}
```

#### Dictionaries/Maps

```json
// dict[K, V] or Mapping[K, V]
{"type": "object", "additionalProperties": <schema(V)>}

// dict without type args
{"type": "object", "additionalProperties": {"type": "string"}}
```

### Enum and Literal Types

#### Literal Values

```json
// Literal["a", "b", "c"]
{"type": "string", "enum": ["a", "b", "c"]}

// Literal[1, 2, 3]
{"type": "integer", "enum": [1, 2, 3]}

// Mixed types - no type field
{"enum": ["a", 1, true]}
```

#### Enum Classes

```json
// enum with string values
{"type": "string", "enum": ["value1", "value2"]}

// enum with integer values
{"type": "integer", "enum": [1, 2, 3]}
```

### Union Types

```json
// Union[X, Y] or X | Y
{"oneOf": [<schema(X)>, <schema(Y)>]}

// Optional[T] (Union[T, None]) - unwrap to just T
<schema(T)>
```

### Structured Types

#### TypedDict / Typed Objects

```json
{
    "type": "object",
    "properties": {
        "field1": <schema(type1)>,
        "field2": <schema(type2)>
    },
    "required": ["field1"]
}
```

#### Dataclasses / Structured Classes

```json
{
    "type": "object",
    "properties": {
        "field1": <schema(type1)>,
        "field2": <schema(type2)>
    },
    "required": ["fields_without_defaults"]
}
```

#### Model Classes (e.g., Pydantic)

```json
{
    "type": "object",
    "properties": {
        "field1": <schema(type1)>,
        "field2": <schema(type2)>
    },
    "required": ["required_fields"]
}
```

### Fallback

Any unrecognized type defaults to `{"type": "string"}`.

## Handling Annotated Types

When encountering `Annotated[T, metadata...]`, unwrap to extract the base type `T` and process it.

## Implementation Notes

### Parameter Description Generation

Generate descriptions by combining:
- Parameter name
- Type information
- Any metadata from annotations

Example: `"Parameter location of type str"`

### Handling Special Parameters

Skip these parameter kinds:
- `*args` (VAR_POSITIONAL)
- `**kwargs` (VAR_KEYWORD)

### Required vs Optional

A parameter is **required** if it has no default value in the function signature.

## Example

### Input Function (Python)

```python
def get_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get weather information for a location."""
    ...
```

### Output Tool Definition

```json
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Parameter location of type str"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Parameter unit of type Literal['celsius', 'fahrenheit']"
                }
            },
            "required": ["location"]
        }
    }
}
```
