---
title: Quickstart
description: Install any-llm and make your first API call in 5 minutes
---

## Requirements

- Python 3.11 or newer
- API keys for your chosen LLM provider

## Installation

```bash
pip install any-llm-sdk[all]  # Install with all provider support
```

### Installing Specific Providers

If you want to install a specific provider from our [supported providers](/any-llm/providers/):

```bash
pip install any-llm-sdk[mistral]  # For Mistral provider
pip install any-llm-sdk[ollama]   # For Ollama provider
# install multiple providers
pip install any-llm-sdk[mistral,ollama]
```

### Library Integration

If you're building a library, install just the base package (`pip install any-llm-sdk`) and let your users install provider dependencies.

> **API Keys:** Set your provider's API key as an environment variable (e.g., `export MISTRAL_API_KEY="your-key"`) or pass it directly using the `api_key` parameter.

## APIs

### Using the AnyLLM Class

For applications making multiple requests with the same provider, use the `AnyLLM` class to avoid repeated provider instantiation:

```python
import os

from any_llm import AnyLLM

# Make sure you have the appropriate API key set
api_key = os.environ.get('MISTRAL_API_KEY')
if not api_key:
    raise ValueError("Please set MISTRAL_API_KEY environment variable")

llm = AnyLLM.create("mistral")

response = llm.completion(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

metadata = llm.get_provider_metadata()
print(f"Supports streaming: {metadata.streaming}")
print(f"Supports tools: {metadata.completion}")
```

### API Call

```python
import os

from any_llm import completion

# Make sure you have the appropriate API key set
api_key = os.environ.get('MISTRAL_API_KEY')
if not api_key:
    raise ValueError("Please set MISTRAL_API_KEY environment variable")

# Recommended: separate provider and model parameters
response = completion(
    model="mistral-small-latest",
    provider="mistral",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### When to Choose Which Approach

**Use Direct API Functions (`completion`, `acompletion`) when:**

- Making simple, one-off requests
- Prototyping or writing quick scripts
- You want the simplest possible interface

**Use Provider Class (`AnyLLM.create`) when:**

- Building applications that make multiple requests with the same provider
- You want to avoid repeated provider instantiation overhead

**Finding model names:** Check the [providers page](/any-llm/providers/) for provider IDs, or use the [`list_models`](/any-llm/api/list-models/) API to see available models for your provider.

## Streaming

For the [providers that support streaming](/any-llm/providers/), you can enable it by passing `stream=True`:

```python
output = ""
for chunk in completion(
    model="mistral-small-latest",
    provider="mistral",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
):
    chunk_content = chunk.choices[0].delta.content or ""
    print(chunk_content)
    output += chunk_content
```

## Reasoning

For [providers that support reasoning](/any-llm/providers/), you can request thinking traces alongside the response using `reasoning_effort`:

```python
from any_llm import completion

response = completion(
    model="claude-sonnet-4-5-20250929",
    provider="anthropic",
    messages=[{"role": "user", "content": "How many r's are in strawberry?"}],
    reasoning_effort="high",
)

# Access the model's thinking trace
if response.choices[0].message.reasoning:
    print(response.choices[0].message.reasoning.content)

# The final answer
print(response.choices[0].message.content)
```

Reasoning also works with streaming — each chunk may include `chunk.choices[0].delta.reasoning`.

## Embeddings

`embedding` and `aembedding` allow you to create vector embeddings from text using the same unified interface across providers.

Not all providers support embeddings - check the [providers documentation](/any-llm/providers/) to see which ones do.

```python
from any_llm import embedding

result = embedding(
    model="text-embedding-3-small",
    provider="openai",
    inputs="Hello, world!" # can be either string or list of strings
)

# Access the embedding vector
embedding_vector = result.data[0].embedding
print(f"Embedding vector length: {len(embedding_vector)}")
print(f"Tokens used: {result.usage.total_tokens}")
```

## Tools

`any-llm` supports tool calling for providers that support it. You can pass a list of tools where each tool is either:

1. **Python callable** - Functions with proper docstrings and type annotations
2. **OpenAI Format tool dict** - Already in OpenAI tool format

```python
from any_llm import completion

def get_weather(location: str, unit: str = "F") -> str:
    """Get weather information for a location.

    Args:
        location: The city or location to get weather for
        unit: Temperature unit, either 'C' or 'F'

    Returns:
        Current weather description
    """
    return f"Weather in {location} is sunny and 75{unit}!"

response = completion(
    model="mistral-small-latest",
    provider="mistral",
    messages=[{"role": "user", "content": "What's the weather in Pittsburgh PA?"}],
    tools=[get_weather]
)
```

any-llm automatically converts your Python functions to OpenAI tools format. Functions must have:
- A docstring describing what the function does
- Type annotations for all parameters
- A return type annotation

## Provider Configuration

Use `client_args` to pass provider-specific arguments to the underlying SDK client during initialization. This is useful for custom timeouts, headers, or provider-specific options.

```python
from any_llm import completion

response = completion(
    model="mistral-small-latest",
    provider="mistral",
    messages=[{"role": "user", "content": "Hello!"}],
    client_args={"timeout": 120},
)
```

### AWS Bedrock: Custom boto3 Client

For AWS Bedrock, you can pass a pre-built `boto3` client. This is useful when you have custom credential configurations (IAM roles, STS assumed roles, named profiles, etc.) that are easier to set up on a boto3 client directly:

```python
import boto3
from any_llm import completion

my_client = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",
)

response = completion(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    provider="bedrock",
    messages=[{"role": "user", "content": "Hello!"}],
    client_args={"client": my_client},
)
```

When a custom `client` is provided, any-llm skips credential verification and uses the client directly.

## Exception Handling

The `any-llm` package provides a unified exception hierarchy that works consistently across all LLM providers.

### Enabling Unified Exceptions

:::note[Opt-in Feature]
Unified exception handling is currently **opt-in**. Set the `ANY_LLM_UNIFIED_EXCEPTIONS` environment variable to enable it:
:::

```bash
export ANY_LLM_UNIFIED_EXCEPTIONS=1
```

When enabled, provider-specific exceptions are automatically converted to `any-llm` exception types. When disabled (default), the original provider exceptions are raised with a deprecation warning.

### Basic Usage

```python
from any_llm import completion
from any_llm.exceptions import (
    RateLimitError,
    AuthenticationError,
    ProviderError,
    AnyLLMError,
)

try:
    response = completion(
        model="gpt-4",
        provider="openai",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except RateLimitError as e:
    print(f"Rate limited: {e.message}")
except AuthenticationError as e:
    print(f"Auth failed: {e.message}")
except ProviderError as e:
    print(f"Provider error: {e.message}")
except AnyLLMError as e:
    print(f"Error: {e.message}")
```

### Accessing Original Exceptions

All unified exceptions preserve the original provider exception for debugging:

```python
from any_llm.exceptions import RateLimitError

messages = [{"role": "user", "content": "Hello!"}]

try:
    response = completion(model="gpt-4", provider="openai", messages=messages)
except RateLimitError as e:
    print(f"Provider: {e.provider_name}")
    print(f"Original exception: {type(e.original_exception)}")
```
