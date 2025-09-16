<p align="center">
  <picture>
    <img src="./images/any-llm-logo.png" width="20%" alt="Project logo"/>
  </picture>
</p>

`any-llm` is a Python library providing a single interface to different llm providers.

### Demo

Try `any-llm` in action with our interactive chat demo that showcases streaming completions and provider switching:

**[📂 Run the Demo](https://github.com/mozilla-ai/any-llm/tree/main/demos/chat#readme)**

The demo features real-time streaming responses, multiple provider support, and collapsible "thinking" content display.

### Getting Started

Refer to the [Quickstart](./quickstart.md) for instructions on installation and usage.

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
