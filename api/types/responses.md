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

## Internal Types

### `ResponsesParams`

Normalized parameters for the Responses API, used internally to pass structured parameters from the public API to provider implementations.

**Import:** `from any_llm.types.responses import ResponsesParams`

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Model identifier (e.g., 'mistral-small-latest') |
| `input` | `str \| list[EasyInputMessageParam \| Message \| ResponseOutputMessageParam \| ResponseFileSearchToolCallParam \| ResponseComputerToolCallParam \| ComputerCallOutput \| ResponseFunctionWebSearchParam \| ResponseFunctionToolCallParam \| FunctionCallOutput \| ToolSearchCall \| ResponseToolSearchOutputItemParamParam \| ResponseReasoningItemParam \| ResponseCompactionItemParamParam \| ImageGenerationCall \| ResponseCodeInterpreterToolCallParam \| LocalShellCall \| LocalShellCallOutput \| ShellCall \| ShellCallOutput \| ApplyPatchCall \| ApplyPatchCallOutput \| McpListTools \| McpApprovalRequest \| McpApprovalResponse \| McpCall \| ResponseCustomToolCallOutputParam \| ResponseCustomToolCallParam \| ItemReference]` | The input payload accepted by provider's Responses API. For OpenAI-compatible providers, this is typically a list mixing text, images, and tool instructions, or a dict per OpenAI spec. |
| `instructions` | `str \| None` |  |
| `max_tool_calls` | `int \| None` |  |
| `text` | `Any \| None` |  |
| `tools` | `list[dict[str, Any]] \| None` | List of tools for tool calling. Should be converted to OpenAI tool format dicts |
| `tool_choice` | `str \| dict[str, Any] \| None` | Controls which tools the model can call |
| `temperature` | `float \| None` | Controls randomness in the response (0.0 to 2.0) |
| `top_p` | `float \| None` | Controls diversity via nucleus sampling (0.0 to 1.0) |
| `max_output_tokens` | `int \| None` | Maximum number of tokens to generate |
| `response_format` | `dict[str, Any] \| type[BaseModel] \| None` | Format specification for the response |
| `stream` | `bool \| None` | Whether to stream the response |
| `parallel_tool_calls` | `bool \| None` | Whether to allow parallel tool calls |
| `top_logprobs` | `int \| None` | Number of top alternatives to return when logprobs are requested |
| `stream_options` | `dict[str, Any] \| None` | Additional options controlling streaming behavior |
| `reasoning` | `dict[str, Any] \| None` | Configuration options for reasoning models. |
| `presence_penalty` | `float \| None` | Penalizes new tokens based on whether they appear in the text so far. |
| `frequency_penalty` | `float \| None` | Penalizes new tokens based on their frequency in the text so far. |
| `truncation` | `str \| None` | Controls how the service truncates the input when it exceeds the model context window. |
| `store` | `bool \| None` | Whether to store the response so it can be retrieved later. |
| `service_tier` | `str \| None` | The service tier to use for this request. |
| `user` | `str \| None` | A unique identifier representing your end user. |
| `metadata` | `dict[str, str] \| None` | Key-value pairs for custom metadata (up to 16 pairs). |
| `previous_response_id` | `str \| None` | The ID of the response to use as the prior turn for this request. |
| `include` | `list[str] \| None` | Items to include in the response (e.g., 'reasoning.encrypted_content'). |
| `background` | `bool \| None` | Whether to run the request in the background and return immediately. |
| `safety_identifier` | `str \| None` | A stable identifier used for safety monitoring and abuse detection. |
| `prompt_cache_key` | `str \| None` | A key to use when reading from or writing to the prompt cache. |
| `prompt_cache_retention` | `str \| None` | How long to retain a prompt cache entry created by this request. |
| `conversation` | `str \| dict[str, Any] \| None` | The conversation to associate this response with (ID string or ConversationParam object). |

## Type Mapping Summary

| Type | Source | Used When |
|------|--------|-----------|
| `ResponseResource` | `openresponses-types` | OpenResponses-compliant providers, non-streaming |
| `Response` | `openai.types.responses` | OpenAI-native providers, non-streaming |
| `ResponseStreamEvent` | `openai.types.responses` | All providers, streaming (`stream=True`) |
| `ResponseInputParam` | `openai.types.responses` | Input parameter type |

For full details on the OpenResponses specification, see the [OpenResponses GitHub repository](https://github.com/openresponsesspec/openresponses). For OpenAI response types, see the [OpenAI Python SDK](https://github.com/openai/openai-python).
