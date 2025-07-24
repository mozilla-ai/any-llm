# Tool Usage in Any LLM

Any LLM provides flexibility in defining tools for interaction. You can either:

1. **Pass a Full Function Schema**: Provide a detailed schema that defines the function's inputs, outputs, and behavior.
2. **Pass a Callable Directly**: Provide a Python callable (e.g., a function or method). Any LLM will automatically convert it into a schema under the hood.

## Passing a Full Function Schema

A function schema is a structured representation of a function, including its name, parameters, and return type. Here's an example:

```python
schema = {
    "name": "add_numbers",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    },
    "return_type": "number"
}

result = any_llm.invoke_tool(schema, {"a": 5, "b": 3})
print(result)  # Output: 8
```

## Passing a Callable Directly

You can pass a Python function directly, and Any LLM will handle the schema generation automatically. For example:

```python
def add_numbers(a: int, b: int) -> int:
    return a + b

result = any_llm.invoke_tool(add_numbers, {"a": 5, "b": 3})
print(result)  # Output: 8
```

Under the hood, Any LLM analyzes the function signature and generates a schema that matches the callable's parameters and return type.

## Benefits

- **Ease of Use**: Developers can focus on writing Python functions without worrying about schema creation.
- **Flexibility**: Advanced users can define custom schemas for more control.

For more details, refer to the [API Documentation](https://github.com/mozilla-ai/any-llm/docs/api).
