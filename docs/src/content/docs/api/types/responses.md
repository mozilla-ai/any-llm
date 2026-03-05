---
title: Responses Types
description: Data models for the OpenResponses API
---

The Responses API types come from two sources depending on the provider:

- **OpenResponses-compliant providers** return `ResponseResource` from the [`openresponses-types`](https://pypi.org/project/openresponses-types/) package.
- **OpenAI-native providers** return `Response` from the `openai` SDK.
- **Streaming** always yields `ResponseStreamEvent` objects.

## Primary Types

### `ResponseResource`

The response object from providers implementing the [OpenResponses specification](https://github.com/openresponsesspec/openresponses).

**Import:** `from openresponses_types import ResponseResource`

**Package:** [`openresponses-types`](https://pypi.org/project/openresponses-types/)

This is the primary return type for OpenResponses-compliant providers. It provides a standardized interface for accessing response content, tool calls, and metadata.

### `Response`

The response object from OpenAI's native Responses API. Re-exported from `openai.types.responses.Response`.

**Import:** `from any_llm.types.responses import Response`

This is returned by providers that use OpenAI's API directly (e.g., the `openai` provider).

### `ResponseStreamEvent`

A single event in a streaming response. Re-exported from `openai.types.responses.ResponseStreamEvent`.

**Import:** `from any_llm.types.responses import ResponseStreamEvent`

Stream events represent incremental updates during response generation, including content deltas, tool call events, and completion signals.

### `ResponseInputParam`

The input type accepted by the `input_data` parameter of `responses()` and `aresponses()`. Re-exported from `openai.types.responses.ResponseInputParam`.

**Import:** `from any_llm.types.responses import ResponseInputParam`

This is typically a list of message items that can include text, images, and tool-related content.

### `ResponseOutputMessage`

An output message within a response. Re-exported from `openai.types.responses.ResponseOutputMessage`.

**Import:** `from any_llm.types.responses import ResponseOutputMessage`

## Type Mapping Summary

| Type | Source | Used When |
|------|--------|-----------|
| `ResponseResource` | `openresponses-types` | OpenResponses-compliant providers, non-streaming |
| `Response` | `openai.types.responses` | OpenAI-native providers, non-streaming |
| `ResponseStreamEvent` | `openai.types.responses` | All providers, streaming (`stream=True`) |
| `ResponseInputParam` | `openai.types.responses` | Input parameter type |

For full details on the OpenResponses specification, see the [OpenResponses GitHub repository](https://github.com/openresponsesspec/openresponses). For OpenAI response types, see the [OpenAI Python SDK](https://github.com/openai/openai-python).
