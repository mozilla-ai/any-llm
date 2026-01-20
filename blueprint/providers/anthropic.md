# API

https://platform.claude.com/docs/en/api/overview

# Differences with OpenAI

- System messages: Extract from messages array and pass via separate `system` parameter (concatenate multiple system messages with newlines)
- Tool role: Convert `role: "tool"` to `role: "user"` with `content: [{type: "tool_result", tool_use_id, content}]`
- Tool calls in assistant: Convert `tool_calls` array to `content: [{type: "tool_use", id, name, input}]`
- Consecutive tool results: Merge into a single user message with multiple `tool_result` blocks
- Image type mapping mapping: `image_url` to `image`
- Base64 data URLs: Parse `data:<media_type>;base64,<data>` into `{type: "base64", media_type, data}`
- Regular URLs: Use `{type: "url", url}`
- Tool spec: `function.parameters` to `input_schema` with `{type: "object", properties, required}`
- Tool choice:
  - `"required"` to `"any"`
  - `{type: "function", function: {name}}` to `{type: "tool", name}`
- Parallel tool calls: `parallel_tool_calls: true` to `disable_parallel_tool_use: false` (inverted logic)
- Stop reason mapping:
  - `end_turn` to `stop`
  - `max_tokens` to `length`
  - `tool_use` to `tool_calls`
- Content blocks:
  - `text` blocks: concatenate into `content` string
  - `tool_use` blocks: `tool_calls` array with `{id, type: "function", function: {name, arguments}}`
  - `thinking` blocks: `reasoning.content`
- Usage: `input_tokens` to `prompt_tokens`, `output_tokens` to `completion_tokens`
- Streaming Event types to handle:
  - `ContentBlockStartEvent`: Initialize content/tool_call/reasoning delta
  - `ContentBlockDeltaEvent`: `text_delta` to content, `input_json_delta` to tool arguments, `thinking_delta` to reasoning
  - `ContentBlockStopEvent`: Set `finish_reason: "tool_calls"` for tool_use blocks
  - `MessageStopEvent`: Set `finish_reason: "stop"`, extract final usage
- Reasoning effort mapping (to `thinking.budget_tokens`):
  - `"none"` to `{type: "disabled"}`
  - `"minimal"` to 1024, `"low"` to 2048, `"medium"` to 8192, `"high"` to 24576, `"xhigh"` to 32768

## Required Parameters

- `max_tokens` is required (set default if not provided, e.g., 8192)

## Unsupported Parameters

- `response_format`: Not supported (raise error with guidance to use tool-based JSON extraction). Until it is out of beta (https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
