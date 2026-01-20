# any-llm-py: Python Implementation Guidelines

- Methods should be async first.
    All the sync methods must be a simple wrapper of the corresponding async method.
- Don't write module-level docstrings.
