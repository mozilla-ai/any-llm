# any-llm

`any-llm` is a Python library providing a single interface to different llm providers.

## Requirements

- Python 3.11 or newer

## Installation

Install `any-llm` with the name of the provider that you plan to use

```bash
pip install any-llm-sdk[mistral,ollama]
```

Refer to [pyproject.toml](https://github.com/mozilla-ai/any-llm/blob/main/pyproject.toml) for a list of the options available.

## Quickstart

### Basic Usage

The primary function is [`completion`][any_llm.completion], which uses a unified interface across all providers:

```python
from any_llm import completion
import os

# Make sure you have the appropriate environment variable set
assert os.environ.get('MISTRAL_API_KEY')
# Basic completion
response = completion(
    model="mistral/mistral-small-latest", # <provider>/<model-id>
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Parameters

For a complete list of available parameters, see the [completion API documentation](./api/completion.md).

### Error Handling

`any-llm` provides custom exceptions to indicate common errors like missing API keys
and parameters that are unsupported by a specific provider.

For more details on exceptions, see the [exceptions API documentation](./api/exceptions.md).

## For AI Systems

This documentation is available in two AI-friendly formats:

- **[llms.txt](https://mozilla-ai.github.io/any-llm/llms.txt)** - A structured overview with curated links to key documentation sections
- **[llms-full.txt](https://mozilla-ai.github.io/any-llm/llms-full.txt)** - Complete documentation content concatenated into a single file
