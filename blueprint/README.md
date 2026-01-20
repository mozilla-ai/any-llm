# any-llm: Language-Agnostic Blueprint

This document defines the core concepts and architecture of `any-llm`, a unified interface for interacting with multiple LLM (Large Language Model) providers. 

Use this blueprint to implement the library in any programming language.

## Overview

**any-llm** provides a single, unified API to communicate with any LLM provider (OpenAI, Anthropic, Mistral, Gemini, etc.), including local ones (llama.cpp, llamafile, ollama), without changing application code. 

The library abstracts provider-specific differences while following the **OpenAI API format as the canonical interface**.

## AnyLLM

The core abstract base class.

```py
from any_llm import AnyLLM, Providers
from any_llm.errors import (
  AuthenticationError,
  AnyLLMError,
)
# Directly pass an API key
# Or use environment variables (MISTRAL_API_KEY, ANTHROPIC_API_KEY, etc.)
llm = AnyLLM.create(Providers.MISTRAL)

try:
  response = llm.completion(
      model="mistral-small-latest",
      messages=[{"role": "user", "content": "Hello!"}]
  )
  print(response.choices[0].message.content)
except AuthenticationError as e:
  print(f"Auth failed: {e.message}")
except AnyLLMError as e:
  print(f"Error: {e.message}")
```

See [any_llm.md](./api/any_llm.md)

## AnyLLMError

Provider-specific errors must be converted to unified any-llm error types.

See [api/errors.md](./api/errors.md)

## Providers

Each provider must subclass AnyLLM and implement the abstract methods.

### SDKs

When available, use the official provider SDK.

If a provider doesn't support one of the methods, check the language-specific specification on how to implement
the lack of support.

If there is an official OpenAI SDK for the language, it must be included as a required dependency.
Any other SDKS must be included as optional dependencies.

The inputs and outputs of the SDKs must be mapped to OpenAI types.
The raw provider responses must be included in an field extending the original OpenAI types.

### OpenAI base

All input/output types must be imported from the official SDK. If there is no official SDK,
implement the input/output types following the API documentation (i.e. [completions.md](./completions.md)).

If a provider offers an OpenAI-compatible API, inherit from the OpenAI provider and adjust the default
parameters as needed (i.e. `api_base`).

### List of Providers to implement

- [OpenAI](./providers/openai.md)
- [Anthropic](./providers/anthropic.md)
- [Databricks](./providers/databricks.md)
- [Gemini](./providers/gemini.md)

## Testing

The OpenAI provider (and those using OpenAI-compatible APIs) must be tested
against the [Acceptance Tests](./acceptance-tests.md).

For providers using their own SDKs, the internal conversion method must be
fully covered by unit tests.
