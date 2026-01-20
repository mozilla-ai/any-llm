# any-llm-ts: TypeScript Implementation Guidelines

- Create a `CompletionOptions` interface with camelCase properties for a more TypeScript-friendly API.
- Implement lazy loading of providers using dynamic imports (`await import()`).
- Use TypeScript's `AsyncIterable` and `async *` generator syntax for streaming responses.
- Use https://github.com/openai/openai-node