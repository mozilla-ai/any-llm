# One interface. Every LLM.

any-llm is a Python library providing a single interface to different LLM providers.

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

[**Get Started**](quickstart.md) | [**View on GitHub**](https://github.com/mozilla-ai/any-llm)

## Why any-llm

<table data-view="cards">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th data-hidden data-card-target data-type="content-ref"></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Switch providers in one line</strong></td>
      <td>Change from OpenAI to Anthropic, Mistral, or any other provider with a single parameter change.</td>
      <td><a href="providers.md">Providers</a></td>
    </tr>
    <tr>
      <td><strong>Unified exception handling</strong></td>
      <td>Consistent error handling across all providers with a unified exception hierarchy.</td>
      <td><a href="quickstart.md">Quickstart</a></td>
    </tr>
    <tr>
      <td><strong>Simple API, powerful features</strong></td>
      <td>Streaming, tool calling, embeddings, reasoning, and more, all through one interface.</td>
      <td><a href="api/any-llm.md">API Reference</a></td>
    </tr>
  </tbody>
</table>

## API Documentation

`any-llm` provides two main interfaces:

**Direct API Functions** (recommended for simple use cases):
- [completion](api/completion.md) - Chat completions with any provider
- [embedding](api/embedding.md) - Text embeddings
- [moderation](api/moderation.md) - Content moderation
- [responses](api/responses.md) - [OpenResponses](https://www.openresponses.org/) API for agentic AI systems

**AnyLLM Class** (recommended for advanced use cases):
- [Provider API](api/any-llm.md) - Lower-level provider interface with metadata access and reusability

## For AI Systems

This documentation is available in two AI-friendly formats:

- **[llms.txt](https://mozilla-ai.github.io/any-llm/llms.txt)** - A structured overview with curated links to key documentation sections
- **[llms-full.txt](https://mozilla-ai.github.io/any-llm/llms-full.txt)** - Complete documentation content concatenated into a single file
