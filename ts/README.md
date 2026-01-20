# any-llm TypeScript Implementation

A unified interface for interacting with multiple LLM providers in TypeScript/JavaScript.

## Installation

```bash
npm install any-llm-ts
```

### Optional Provider SDKs

Install the SDKs for the providers you want to use:

```bash
# For Anthropic
npm install @anthropic-ai/sdk

# For Google Gemini
npm install @google/generative-ai

# For Mistral
npm install @mistralai/mistralai
```

## Quick Start

```typescript
import { AnyLLM, Providers } from "any-llm";

// Create a provider instance
// Or use environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
const llm = await AnyLLM.create("openai", { apiKey: "sk-..." });

// Or use environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
const llm = await AnyLLM.create("openai");

// Make a completion request
const response = await llm.completion({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello!" }],
});

console.log(response.choices[0].message.content);
```

## Streaming

```typescript
const stream = llm.completion({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Tell me a story" }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

## Error Handling

All provider-specific errors are mapped to unified error types:

```typescript
import {
  AnyLLMError,
  RateLimitError,
  AuthenticationError,
  InvalidRequestError,
  ModelNotFoundError,
  ContextLengthExceededError,
  ContentFilterError,
} from "any-llm";

try {
  await llm.completion({ model: "gpt-4o", messages: [...] });
} catch (error) {
  if (error instanceof RateLimitError) {
    // Handle rate limiting
  } else if (error instanceof AuthenticationError) {
    // Handle auth errors
  }
}
```

## Tool Calling

```typescript
const response = await llm.completion({
  model: "gpt-4o",
  messages: [{ role: "user", content: "What's the weather in Paris?" }],
  tools: [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get current weather for a location",
        parameters: {
          type: "object",
          properties: {
            location: { type: "string", description: "City name" },
          },
          required: ["location"],
        },
      },
    },
  ],
});

if (response.choices[0].message.tool_calls) {
  // Handle tool calls
}
```

## Structured Output

```typescript
const response = await llm.completion({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Extract the name and age" }],
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "person",
      schema: {
        type: "object",
        properties: {
          name: { type: "string" },
          age: { type: "number" },
        },
        required: ["name", "age"],
      },
    },
  },
});
```
