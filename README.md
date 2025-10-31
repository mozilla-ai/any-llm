<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/mozilla-ai/any-llm/refs/heads/main/docs/images/any-llm-logo-mark.png" width="20%" alt="Project logo"/>
  </picture>
</p>

<div align="center">

# any-llm

[![Read the Blog Post](https://img.shields.io/badge/Read%20the%20Blog%20Post-red.svg)](https://blog.mozilla.ai/introducing-any-llm-a-unified-api-to-access-any-llm-provider/)

[![Docs](https://github.com/mozilla-ai/any-llm/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/any-llm/actions/workflows/docs.yaml/)

[![Linting](https://github.com/mozilla-ai/any-llm/actions/workflows/lint.yaml/badge.svg)](https://github.com/mozilla-ai/any-llm/actions/workflows/lint.yaml/)
[![Unit Tests](https://github.com/mozilla-ai/any-llm/actions/workflows/tests-unit.yaml/badge.svg)](https://github.com/mozilla-ai/any-llm/actions/workflows/tests-unit.yaml/)
[![Integration Tests](https://github.com/mozilla-ai/any-llm/actions/workflows/tests-integration.yaml/badge.svg)](https://github.com/mozilla-ai/any-llm/actions/workflows/tests-integration.yaml/)

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/any-llm-sdk)](https://pypi.org/project/any-llm-sdk/)
<a href="https://discord.gg/4gf3zXrQUc">
    <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
</a>

**Communicate with any LLM provider using a single, unified interface.**
Switch between OpenAI, Anthropic, Mistral, Ollama, and more without changing your code.

[Documentation](https://mozilla-ai.github.io/any-llm/) | [Try the Demos](#-try-it) | [Contributing](#-contributing)

</div>

## Quickstart

```python
pip install 'any-llm-sdk[mistral,ollama]'

export MISTRAL_API_KEY="YOUR_KEY_HERE"  # or OPENAI_API_KEY, etc
from any_llm import completion
import os

# Make sure you have the appropriate environment variable set
assert os.environ.get('MISTRAL_API_KEY')

response = completion(
    model="mistral-small-latest",
    provider="mistral",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```
**That's it!** Change the provider name and add provider-specific keys to switch between LLM providers.


## Installation

### Requirements

- Python 3.11 or newer
- API keys for whichever LLM providers you want to use

### Basic Installation

Install support for specific providers:

```bash
pip install 'any-llm-sdk[openai]'           # Just OpenAI
pip install 'any-llm-sdk[mistral,ollama]'   # Multiple providers
pip install 'any-llm-sdk[all]'              # All supported providers
```

See our [list of supported providers](https://mozilla-ai.github.io/any-llm/providers/) to choose which ones you need.

### Setting Up API Keys

Set environment variables for your chosen providers:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export MISTRAL_API_KEY="your-key-here"
# ... etc
```

Alternatively, pass API keys directly in your code (see [Usage](#usage) examples).

## Why choose `any-llm`?

- **Simple, unified interface** - Single function for all providers, switch models with just a string change
- **Developer friendly** - Full type hints for better IDE support and clear, actionable error messages
- **Leverages official provider SDKs** - Ensures maximum compatibility
- **Stays framework-agnostic** so it can be used across different projects and use cases
- **Battle-tested** - Powers our own production tools ([any-agent](https://github.com/mozilla-ai/any-agent))
- **No Proxy or Gateway server required** - Direct connections to whichever LLM provider you need.

## Usage

`any-llm` offers two main approaches for interacting with LLM providers:

#### Option 1: Direct API Functions (Recommended for Bootstrapping and Experimentation)

**Recommended approach:** Use separate `provider` and `model` parameters:

```python
from any_llm import completion
import os

# Make sure you have the appropriate environment variable set
assert os.environ.get('MISTRAL_API_KEY')

response = completion(
    model="mistral-small-latest",
    provider="mistral",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

**Alternative syntax:** Use combined `provider:model` format:

```python
response = completion(
    model="mistral:mistral-small-latest", # <provider_id>:<model_id>
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### Option 2: AnyLLM Class (Recommended for Production)

For applications that need to reuse providers, perform multiple operations, or require more control:

```python
from any_llm import AnyLLM

llm = AnyLLM.create("mistral", api_key="your-mistral-api-key")

response = llm.completion(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)

```

#### When to Use Which Approach

| Approach | Best For | Connection Handling |
|----------|----------|---------------------|
| **Direct API Functions** (`completion`) | Scripts, notebooks, single requests | New client per call (stateless) |
| **AnyLLM Class** (`AnyLLM.create`) | Production apps, multiple requests | Reuses client (connection pooling) |

Both approaches support identical features : streaming, tools, responses API, etc.

### Responses API

For providers that implement the OpenAI-style Responses API, use [`responses`](https://mozilla-ai.github.io/any-llm/api/responses/) or `aresponses`:

```python
from any_llm import responses

result = responses(
    model="gpt-4o-mini",
    provider="openai",
    input_data=[
        {"role": "user", "content": [
            {"type": "text", "text": "Summarize this in one sentence."}
        ]}
    ],
)

# Non-streaming returns an OpenAI-compatible Responses object alias
print(result.output_text)
```

### Finding the Right Model

The `provider_id` should match our [supported provider names](https://mozilla-ai.github.io/any-llm/providers/).

The `model_id` is passed directly to the provider. To find available models:
- Check the provider's documentation
- Use our `list_models` API (if the provider supports it)


## Try It

Try `any-llm` in action with our interactive demos:

### 💬 Chat Demo
**[📂 Run the Chat Demo](./demos/chat/README.md)**

An interactive chat interface showcasing streaming completions and provider switching:
- Real-time streaming responses
- Easy switchign between multiple LLM providers
- Collapsible "thinking" content display for supported models
- Auto-scrolling chat interface

### 🔍 Model Finder Demo
**[📂 Run the Model Finder Demo](./demos/finder/README.md)**

A model discovery tool featuring:
- Search and filter models across all your configured providers
- Provider status dashboard
- API configuration checker

## Motivation

The landscape of LLM provider interfaces is fragmented. While OpenAI's API has become the de facto standard, providers implement slight variations in parameter names, response formats, and feature sets. This creates a need for light wrappers that gracefully handle these differences while maintaining a consistent interface.

**Existing Solutions and Their Limitations:**

- **[LiteLLM](https://github.com/BerriAI/litellm)**: Popular but reimplements provider interfaces rather than leveraging official SDKs, leading to potential compatibility issues.
- **[AISuite](https://github.com/andrewyng/aisuite/issues)**: Clean, modular approach but lacks active maintenance, comprehensive testing, and modern Python typing standards.
- **[Framework-specific solutions](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/models)**: Some agent frameworks either depend on LiteLLM or implement their own provider integrations, creating fragmentation
- **[Proxy Only Solutions](https://openrouter.ai/)**: solutions like [OpenRouter](https://openrouter.ai/) and [Portkey](https://github.com/Portkey-AI/portkey-python-sdk) require a hosted proxy between your code and the LLM provider.

`any-llm` addresses these challenges by leveraging official SDKs when available, maintaining framework-agnostic design, and requiring no proxy servers.

## Documentation
- **[Full Documentation](https://mozilla-ai.github.io/any-llm/)** - Complete guides and API reference
- **[Supported Providers](https://mozilla-ai.github.io/any-llm/providers/)** - List of all supported LLM providers
- **[Cookbook Examples](https://mozilla-ai.github.io/any-llm/cookbook/)** - In-depth usage examples


## Contributing
We welcome contributions from developers of all skill levels! Please see our [Contributing Guide](CONTRIBUTING.md) or open an issue to discuss changes.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.md) file for details.
