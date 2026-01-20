# any-llm-py: Python Implementation Guidelines

- Methods should be async first.
    All the sync methods must be a simple wrapper of the corresponding async method.
    Use the existing wrappers in `py/src/utils/aio.py`.
- Don't write module-level docstrings.
- Use https://github.com/openai/openai-python
