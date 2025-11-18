---
schema:
  type: "HowTo"
  name: "How to Install and Use any-llm"
  description: "Step-by-step guide to installing any-llm and making your first API call with Python"
  totalTime: "PT5M"
  tool:
    - "Python 3.11 or newer"
    - "pip package manager"
  supply:
    - "API key from your chosen LLM provider"
  steps:
    - name: "Install any-llm"
      text: "Install any-llm with your chosen providers using pip. Use the all option to install support for all providers."
      url: "https://mozilla-ai.github.io/any-llm/quickstart/#installation"
    - name: "Set up API keys"
      text: "Configure your provider's API key as an environment variable. Make sure you have the appropriate environment variable set for your chosen provider."
      url: "https://mozilla-ai.github.io/any-llm/quickstart/#api-keys"
    - name: "Make your first completion call"
      text: "Import the completion function from any-llm and create your first API call with your chosen model and provider"
      url: "https://mozilla-ai.github.io/any-llm/quickstart/#your-first-api-call"
---


### Requirements

- Python 3.11 or newer
- API keys for your chosen LLM provider

### Installation
```bash
pip install any-llm-sdk[all]  # Install with all provider support
```

#### Installing Specific Providers

If you want to install a specific provider from our [supported providers](./providers.md):

```bash
pip install any-llm-sdk[mistral]  # For Mistral provider
pip install any-llm-sdk[ollama]   # For Ollama provider
# install multiple providers
pip install any-llm-sdk[mistral,ollama]
```

#### Library Integration

If you're building a library, install just the base package (`pip install any-llm-sdk`) and let your users install provider dependencies.

> **API Keys:** Set your provider's API key as an environment variable (e.g., `export MISTRAL_API_KEY="your-key"`) or pass it directly using the `api_key` parameter.

### Your First API Call 

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

### Advanced: Using the AnyLLM Class

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

#### When to Choose Which Approach

**Use Direct API Functions (`completion`, `acompletion`) when:**

- Making simple, one-off requests
- Prototyping or writing quick scripts
- You want the simplest possible interface

**Use Provider Class (`AnyLLM.create`) when:**

- Building applications that make multiple requests with the same provider
- You want to avoid repeated provider instantiation overhead

**Finding model names:** Check the [providers page](./providers.md) for provider IDs, or use the [`list_models`](./api/list_models.md) API to see available models for your provider.

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
