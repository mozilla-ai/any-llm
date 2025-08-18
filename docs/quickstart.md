## Quickstart

### Requirements

- Python 3.11 or newer
- API_KEYS to access to whichever LLM you choose to use.

### Installation

#### Direct Usage

In your pip install, include the [supported providers](./providers.md) that you plan on using, or use the `all` option if you want to install support for all `any-llm` supported providers.

```bash
pip install any-llm-sdk[mistral]  # For Mistral provider
pip install any-llm-sdk[ollama]   # For Ollama provider
# install multiple providers
pip install any-llm-sdk[mistral,ollama]
# or install support for all providers
pip install any-llm-sdk[all]
```

#### Library Integration

If you're integrating `any-llm` into your own library that others will use, you only need to install the base package:

```bash
pip install any-llm-sdk
```

In this scenario, the end users of your library will be responsible for installing the appropriate provider dependencies when they want to use specific providers. `any-llm` is designed so that you'll only encounter exceptions at runtime if you try to use a provider without having the required dependencies installed.

Those exceptions will clearly describe what needs to be installed to resolve the issue.

Make sure you have the appropriate API key environment variable set for your provider. Alternatively,
you could use the `api_key` parameter when making a completion call instead of setting an environment variable.

```bash
export MISTRAL_API_KEY="YOUR_KEY_HERE"  # or OPENAI_API_KEY, etc
```

### Basic Usage

[`completion`][any_llm.completion] and [`acompletion`][any_llm.acompletion] use a unified interface across all providers.

**Recommended approach:** Use separate `provider` and `model` parameters:

```python
import os

from any_llm import completion, ProviderName

# Make sure you have the appropriate environment variable set
assert os.environ.get('MISTRAL_API_KEY')

# Recommended: separate provider and model parameters
response = completion(
    model="mistral-small-latest",
    provider="mistral", # or ProviderName.MISTRAL
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

**Alternative syntax:** You can also use the combined `provider:model` format:

```python
response = completion(
    model="mistral:mistral-small-latest",  # <provider_id>:<model_id>
    messages=[{"role": "user", "content": "Hello!"}]
)
```

The provider_id should be specified according to the [provider ids supported by any-llm](./providers.md).
The `model_id` portion is passed directly to the provider internals: to understand what model ids are available for a provider,
you will need to refer to the provider documentation or use our [`list_models`](./api/list_models.md)  API if the provider supports that API.

### Streaming

For the [providers that support streaming](./providers.md), you can enable it by passing `stream=True`:

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

### Embeddings

[`embedding`][any_llm.embedding] and [`aembedding`][any_llm.aembedding] allow you to create vector embeddings from text using the same unified interface across providers.

Not all providers support embeddings - check the [providers documentation](./providers.md) to see which ones do.

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

### Tools

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
