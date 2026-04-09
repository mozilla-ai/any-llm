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

- **Switch providers in one line**: Change from OpenAI to Anthropic, Mistral, or any other provider with a single parameter change.
- **Unified exception handling**: Consistent error handling across all providers with a unified exception hierarchy.
- **Simple API, powerful features**: Streaming, tool calling, embeddings, reasoning, and more, all through one interface.

## API Documentation

`any-llm` provides two main interfaces:

**Direct API Functions** (recommended for simple use cases):
- [completion](api/completion.md) - Chat completions with any provider
- [embedding](api/embedding.md) - Text embeddings
- [responses](api/responses.md) - [OpenResponses](https://www.openresponses.org/) API for agentic AI systems

**AnyLLM Class** (recommended for advanced use cases):
- [Provider API](api/any-llm.md) - Lower-level provider interface with metadata access and reusability

## For AI Systems

This documentation is available in two AI-friendly formats:

- **[llms.txt](https://mozilla-ai.github.io/any-llm/llms.txt)** - A structured overview with curated links to key documentation sections
- **[llms-full.txt](https://mozilla-ai.github.io/any-llm/llms-full.txt)** - Complete documentation content concatenated into a single file
