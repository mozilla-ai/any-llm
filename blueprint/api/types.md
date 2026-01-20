# Type Definitions

This document describes how to handle type definitions across language implementations.

## Type Re-export Strategy

For languages with an official OpenAI SDK:

- Re-export all relevant types from the OpenAI SDK through the `any-llm` library to provide a unified import experience.


For languages that don't have an official OpenAI SDK:

- Define types from scratch following the OpenAI API specification in [completions.md](./completions.md)
- Maintain compatibility with the API schema exactly as documented
- Use language-appropriate idioms for naming and structure

## Additional Types

In addition to re-exporting OpenAI SDK types, the library should define its own types for:

### Options/Parameters Interface

Create a convenience interface/type for completion options that follows the language's naming conventions while mapping to the OpenAI API parameters.

The provider implementation should handle converting these language-idiomatic names to the actual API parameter names.
