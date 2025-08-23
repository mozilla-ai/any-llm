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

A single interface to use different llm providers.

</div>

## [Documentation](https://mozilla-ai.github.io/any-llm/)

## [Supported Providers](https://mozilla-ai.github.io/any-llm/providers)

## Key Features

`any-llm` offers:
- **Simple, unified interface** - one function for all providers, switch models with just a string change
- **Developer friendly** - full type hints for better IDE support and clear, actionable error messages
- **Leverages official provider SDKs** when available, reducing maintenance burden and ensuring compatibility
- **Stays framework-agnostic** so it can be used across different projects and use cases
- **Actively maintained** - we use this in our own product ([any-agent](https://github.com/mozilla-ai/any-agent)) ensuring continued support
- **No Proxy or Gateway server required** so you don't need to deal with setting up any other service to talk to whichever LLM provider you need.

## Motivation

The landscape of LLM provider interfaces presents a fragmented ecosystem with several challenges that `any-llm` aims to address:

**The Challenge with API Standardization:**

While the OpenAI API has become the de facto standard for LLM provider interfaces, providers implement slight variations. Some providers are fully OpenAI-compatible, while others may have different parameter names, response formats, or feature sets. This creates a need for light wrappers that can gracefully handle these differences while maintaining a consistent interface.

**Existing Solutions and Their Limitations:**

- **[LiteLLM](https://github.com/BerriAI/litellm)**: While popular, it reimplements provider interfaces rather than leveraging official SDKs, which can lead to compatibility issues and unexpected behavior modifications
- **[AISuite](https://github.com/andrewyng/aisuite/issues)**: Offers a clean, modular approach but lacks active maintenance, comprehensive testing, and modern Python typing standards.
- **[Framework-specific solutions](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/models)**: Some agent frameworks either depend on LiteLLM or implement their own provider integrations, creating fragmentation
- **[Proxy Only Solutions](https://openrouter.ai/)**: solutions like [OpenRouter](https://openrouter.ai/) and [Portkey](https://github.com/Portkey-AI/portkey-python-sdk) require a hosted proxy to serve as the interface between your code and the LLM provider.

## Demo

Try `any-llm` in action with our interactive chat demo that showcases streaming completions and provider switching:

**[📂 Run the Demo](./demos/chat/README.md)**

The demo features:
- Real-time streaming responses with character-by-character display
- Support for multiple LLM providers with easy switching
- Collapsible "thinking" content display for supported models
- Clean chat interface with auto-scrolling

## Quickstart

### Requirements

- Python 3.11 or newer
- API_KEYS to access to whichever LLM you choose to use.

### Installation

In your pip install, include the [supported providers](https://mozilla-ai.github.io/any-llm/providers/) that you plan on using, or use the `all` option if you want to install support for all `any-llm` supported providers.

```bash
pip install 'any-llm-sdk[mistral,ollama]'
```

Make sure you have the appropriate API key environment variable set for your provider. Alternatively,
you could use the `api_key` parameter when making a completion call instead of setting an environment variable.

```bash
export MISTRAL_API_KEY="YOUR_KEY_HERE"  # or OPENAI_API_KEY, etc
```

### Basic Usage

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

**Alternative syntax:** You can also use the combined `provider:model` format:

```python
response = completion(
    model="mistral:mistral-small-latest", # <provider_id>:<model_id>
    messages=[{"role": "user", "content": "Hello!"}]
)
```

The provider_id should be specified according to the [provider ids supported by any-llm](https://mozilla-ai.github.io/any-llm/providers/).
The `model_id` portion is passed directly to the provider internals: to understand what model ids are available for a provider,
you will need to refer to the provider documentation or use our `list_models` API if the provider supports that API.


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
