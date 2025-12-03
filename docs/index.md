---
schema:
  type: "SoftwareSourceCode"
  name: "any-llm"
  description: "A Python library providing a single interface to different LLM providers including OpenAI, Anthropic, Mistral, and more"
  programmingLanguage: "Python"
  codeRepository: "https://github.com/mozilla-ai/any-llm"
  license: "https://github.com/mozilla-ai/any-llm/blob/main/LICENSE"
---

<p align="center">
  <picture>
    <img src="./images/any-llm-logo.png" width="20%" alt="any-llm logo"/>
  </picture>
      <p align="center">  <b>One interface. Every LLM. </b></p>
</p>

`any-llm` is a Python library providing a single interface to different llm providers.

```python
from any_llm import completion

# Using the messages format
response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is Python?"}],
    provider="openai"
)
print(response)

# Switch providers without changing your code
response = completion(
    model="claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": "What is Python?"}],
    provider="anthropic"
)
print(response)
```

### Why any-llm
  - Switch providers in one line
  - Consistent error handling across providers
  - Simple API, powerful features

[View supported providers →](./providers.md)

### Getting Started

**[Get started in 5 minutes →](./quickstart.md)** - Install the library and run your first API call.


### Demo

Try `any-llm` in action with our interactive chat demo:

**[📂 Run the Demo](https://github.com/mozilla-ai/any-llm/tree/main/demos/chat#readme)**

Features: real-time streaming responses, multiple provider support, and collapsible "thinking" content display.

### API Documentation

`any-llm` provides two main interfaces:

**Direct API Functions** (recommended for simple use cases):
- [completion](./api/completion.md) - Chat completions with any provider
- [embedding](./api/embedding.md) - Text embeddings
- [responses](./api/responses.md) - OpenAI-style Responses API

**AnyLLM Class** (recommended for advanced use cases):
- [Provider API](./api/any_llm.md) - Lower-level provider interface with metadata access and reusability

### Error Handling

`any-llm` provides custom exceptions to indicate common errors like missing API keys
and parameters that are unsupported by a specific provider.

For more details on exceptions, see the [exceptions API documentation](./api/exceptions.md).

## For AI Systems

This documentation is available in two AI-friendly formats:

- **[llms.txt](https://mozilla-ai.github.io/any-llm/llms.txt)** - A structured overview with curated links to key documentation sections
- **[llms-full.txt](https://mozilla-ai.github.io/any-llm/llms-full.txt)** - Complete documentation content concatenated into a single file
