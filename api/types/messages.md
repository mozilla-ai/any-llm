---
title: Messages Types
description: Data models for the Anthropic Messages API
---

The Messages API types are Pydantic models used by `any_llm.api.messages()` and `any_llm.api.amessages()`.

## Primary Types

### `MessageResponse`

Full response from the Messages API.

**Import:** `from any_llm.types.messages import MessageResponse`

| Field | Type | Description |
|-------|------|-------------|
| `container` | `BetaContainer \| None` |  |
| `content` | `list[ThinkingBlock \| TextBlock \| ThinkingBlock \| RedactedThinkingBlock \| ToolUseBlock \| ServerToolUseBlock \| WebSearchToolResultBlock \| WebFetchToolResultBlock \| CodeExecutionToolResultBlock \| BashCodeExecutionToolResultBlock \| TextEditorCodeExecutionToolResultBlock \| ToolSearchToolResultBlock \| ContainerUploadBlock \| BetaTextBlock \| BetaThinkingBlock \| BetaRedactedThinkingBlock \| BetaToolUseBlock \| BetaServerToolUseBlock \| BetaWebSearchToolResultBlock \| BetaWebFetchToolResultBlock \| BetaAdvisorToolResultBlock \| BetaCodeExecutionToolResultBlock \| BetaBashCodeExecutionToolResultBlock \| BetaTextEditorCodeExecutionToolResultBlock \| BetaToolSearchToolResultBlock \| BetaMCPToolUseBlock \| BetaMCPToolResultBlock \| BetaContainerUploadBlock \| BetaCompactionBlock \| BetaFallbackBlock]` |  |
| `stop_reason` | `Literal['end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'pause_turn', 'compaction', 'refusal', 'model_context_window_exceeded'] \| None` |  |
| `usage` | `MessageUsage` |  |
| `context_management` | `BetaContextManagementResponse \| None` |  |
| `diagnostics` | `BetaDiagnostics \| None` |  |

### `MessageContentBlock`

Content block in a Messages API response.

**Import:** `from any_llm.types.messages import MessageContentBlock`


### `MessageUsage`

Token usage information for Messages API.

**Import:** `from any_llm.types.messages import MessageUsage`

| Field | Type | Description |
|-------|------|-------------|
| `iterations` | `list[BetaMessageIterationUsage \| BetaCompactionIterationUsage \| BetaAdvisorMessageIterationUsage \| BetaFallbackMessageIterationUsage] \| None` |  |
| `speed` | `Literal['standard', 'fast'] \| None` |  |

### `MessageStreamEvent`

Union of stream event types. Each one subclasses the matching Anthropic SDK `Raw*` event and widens it so beta responses (context management, compaction) round-trip without loss:

- `MessageStartEvent`: `type: 'message_start'`, `message: MessageResponse`
- `MessageDeltaEvent`: `type: 'message_delta'`, `delta: MessageDelta`, `usage: MessageDeltaUsage`, `context_management: BetaContextManagementResponse | None`
- `MessageStopEvent`: `type: 'message_stop'`, `message: MessageResponse | None`
- `ContentBlockStartEvent`: `type: 'content_block_start'`, `index: int`, `content_block: MessageContentBlock`
- `ContentBlockDeltaEvent`: `type: 'content_block_delta'`, `index: int`, `delta: RawContentBlockDelta | CompactionDelta`
- `ContentBlockStopEvent`: `type: 'content_block_stop'`, `index: int`, `content_block: MessageContentBlock | None`

`message` on `MessageStopEvent` and `content_block` on `ContentBlockStopEvent` are populated only by providers with a native Anthropic Messages API; the Chat Completions bridge leaves them unset.

**Import:** `from any_llm.types.messages import MessageStreamEvent`


## Internal Types

### `MessagesParams`

Normalized parameters for the Anthropic Messages API, used internally to pass structured parameters from the public API to provider implementations.

**Import:** `from any_llm.types.messages import MessagesParams`

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Model identifier |
| `messages` | `list[dict[str, Any]]` | List of messages for the conversation |
| `max_tokens` | `int` | Maximum number of tokens to generate (required by Anthropic API) |
| `system` | `str \| list[dict[str, Any]] \| None` | System prompt (string or list of content blocks with optional cache_control) |
| `temperature` | `float \| None` | Controls randomness in the response (0.0 to 1.0) |
| `top_p` | `float \| None` | Controls diversity via nucleus sampling |
| `top_k` | `int \| None` | Only sample from the top K options for each subsequent token |
| `stream` | `bool \| None` | Whether to stream the response |
| `stop_sequences` | `list[str] \| None` | Custom text sequences that will cause the model to stop generating |
| `tools` | `list[dict[str, Any]] \| None` | List of tools in Anthropic format ({name, description, input_schema}) |
| `tool_choice` | `dict[str, Any] \| None` | Controls which tool the model uses |
| `metadata` | `dict[str, Any] \| None` | Request metadata |
| `thinking` | `dict[str, Any] \| None` | Thinking/reasoning configuration |
| `cache_control` | `dict[str, Any] \| None` | Cache control configuration for prompt caching |
| `context_management` | `dict[str, Any] \| None` | Anthropic context management configuration |
| `betas` | `list[str] \| None` | Anthropic beta identifiers |
| `output_format` | `type \| dict[str, Any] \| None` |  |
