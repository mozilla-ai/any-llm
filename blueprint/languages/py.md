# any-llm-py: Python Implementation Guidelines

- Methods should be async first.
    All the sync methods must be a simple wrapper of the corresponding async method.
    Use the existing wrappers in `py/src/utils/aio.py`.
- Don't write module-level docstrings.
- Use https://github.com/openai/openai-python
- Use https://github.com/anthropics/anthropic-sdk-python

## Message Object Conversion

The `_convert_completion_params` method **must** automatically convert any `ChatCompletionMessage` objects (Pydantic models returned in responses) to dictionaries when they appear in the `messages` list.

This is required because the OpenAI Python SDK has an input/output type asymmetry:
- **Input** (`ChatCompletionMessageParam`): A TypeAlias expecting plain dictionaries
- **Output** (`ChatCompletionMessage`): A Pydantic model returned from the API

Users building conversation/agent loops commonly append response messages back to the input list:

```py
messages = [{"role": "user", "content": "Hello"}]
result = await llm.acompletion(model="gpt-4o", messages=messages)
messages.append(result.choices[0].message)  # This is a Pydantic model
# Next call should work without manual conversion:
result = await llm.acompletion(model="gpt-4o", messages=messages)
```

Implementation example:
```py
if "messages" in api_params:
    api_params["messages"] = [
        msg.model_dump() if hasattr(msg, "model_dump") else msg
        for msg in api_params["messages"]
    ]
```
