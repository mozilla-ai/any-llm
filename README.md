<p align="center">
  <picture>
    <img src="docs/images/any-llm-logo-mark.png" width="20%" alt="Project logo"/>
  </picture>
</p>

<div align="center">

# any-llm

[![Docs](https://github.com/mozilla-ai/any-llm/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/any-llm/actions/workflows/docs.yaml/)
[![Unit Tests](https://github.com/mozilla-ai/any-llm/actions/workflows/tests-unit.yaml/badge.svg)](https://github.com/mozilla-ai/any-llm/actions/workflows/tests-unit.yaml/)
[![Integration Tests](https://github.com/mozilla-ai/any-llm/actions/workflows/tests-integration.yaml/badge.svg)](https://github.com/mozilla-ai/any-llm/actions/workflows/tests-integration.yaml/)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

A single interface to use and evaluate different llm providers.

</div>

## [Documentation](https://mozilla-ai.github.io/any-llm/)

## [Supported Providers](https://mozilla-ai.github.io/any-llm/providers)

## Motivation

The landscape of LLM provider interfaces presents a fragmented ecosystem with several challenges that `any-llm` aims to address:

**The Challenge with API Standardization:**

While the OpenAI API has become the de facto standard for LLM provider interfaces, providers implement slight variations. Some providers are fully OpenAI-compatible, while others may have different parameter names, response formats, or feature sets. This creates a need for light wrappers that can gracefully handle these differences while maintaining a consistent interface.

**Existing Solutions and Their Limitations:**

- **[LiteLLM](https://github.com/BerriAI/litellm)**: While popular, it reimplements provider interfaces rather than leveraging official SDKs, which can lead to compatibility issues and unexpected behavior modifications
- **[AISuite](https://github.com/andrewyng/aisuite/issues)**: Offers a clean, modular approach but lacks active maintenance, comprehensive testing, and modern Python typing standards.
- **[Framework-specific solutions](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/models)**: Some agent frameworks either depend on LiteLLM or implement their own provider integrations, creating fragmentation
- **[Proxy Only Solutions](https://openrouter.ai/)**: solutions like OpenRouter require a hosted proxy to serve as the interface between your code and the LLM provider. `any-llm` allows you to communicate directly with the LLM provider without the need for a hosted proxy.

**Our Approach:**

`any-llm` fills the gap by providing a simple, well-maintained interface that:
- **Leverages official provider SDKs** when available, reducing maintenance burden and ensuring compatibility
- **Stays framework-agnostic** so it can be used across different projects and use cases
- **Provides active maintenance** we support this in our product ([any-agent](https://github.com/mozilla-ai/any-agent)) so we're motivated to maintain it.
- **No Proxy or Gateway server required** so you don't need to deal with setting up any other service to talk to whichever LLM provider you need.



## Requirements

- Python 3.11 or newer

## Quickstart

Refer to [pyproject.toml](./pyproject.toml) for a list of the options available.
Update your pip install command to include the frameworks that you plan on using:

```bash
pip install 'any-llm-sdk[mistral]'
```

Make sure you have the appropriate API key environment variable set for your provider

```bash
export MISTRAL_API_KEY="YOUR_KEY_HERE"  # or OPENAI_API_KEY, etc
```

```python
from any_llm import completion

# format for model is "<provider>/<model-name>"
response = completion(
    model="mistral/mistral-small-latest",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```
