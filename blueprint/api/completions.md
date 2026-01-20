# OpenAI Chat Completions API Schema

This document describes the input and output schemas for the OpenAI Chat Completions API endpoint.

## Required Parameters

### messages
- Type: array
- Description: A list of messages comprising the conversation so far. Supports text, images, and audio depending on model.
- Possible types (union):
  - Developer Message:
    - role (required): "developer"
    - content (required): string | array of text parts
    - name (optional): string
  - System Message:
    - role (required): "system"
    - content (required): string | array of text parts
    - name (optional): string
  - User Message:
    - role (required): "user"
    - content (required): string | array of content parts (text, image, audio, file)
    - name (optional): string
  - Assistant Message:
    - role (required): "assistant"
    - content (optional): string | array of text/refusal parts
    - audio (optional): object with id for previous audio response
    - tool_calls (optional): array of tool call objects
    - function_call (deprecated): object with name and arguments
    - name (optional): string
    - refusal (optional): string
  - Tool Message:
    - role (required): "tool"
    - content (required): string | array of text parts
    - tool_call_id (required): string
  - Function Message (deprecated):
    - role (required): "function"
    - content (required): string | null
    - name (required): string

#### Content Part Types
- Text Part:
  - type (required): "text"
  - text (required): string
- Image Part:
  - type (required): "image_url"
  - image_url (required): Either a URL of the image or the base64 encoded image data.
  - detail (optional: auto|low|high)
- Audio Part:
  - type (required): "input_audio"
  - input_audio (required): object with data (base64), format (wav|mp3)
- File Part:
  - type (required): "file"
  - file (required): object with file_id or file_data, filename

### model
- Type: string
- Description: Model ID used to generate the response (e.g., gpt-4o, o3).

## Optional Parameters

### audio
- Type: object | null
- Default: null
- Description: Parameters for audio output. Required when modalities: ["audio"].
- Properties:
  - format (required): "wav" | "aac" | "mp3" | "flac" | "opus" | "pcm16"
  - voice (required): string or one of "alloy" | "ash" | "ballad" | "coral" | "echo" | "sage" | "shimmer" | "verse" | "marin" | "cedar"

### frequency_penalty
- Type: number | null
- Default: 0
- Description: Number between -2.0 and 2.0. Positive values penalize repeated tokens.

### logit_bias
- Type: map
- Default: null
- Description: Map of token IDs to bias values (-100 to 100) to modify token likelihood.

### logprobs
- Type: boolean | null
- Default: false
- Description: Whether to return log probabilities of output tokens.

### max_completion_tokens
- Type: integer | null
- Default: null
- Description: Upper bound for tokens generated (including reasoning tokens).

### metadata
- Type: map
- Default: null
- Description: Up to 16 key-value pairs for storing additional information. Keys max 64 chars, values max 512 chars.

### modalities
- Type: array
- Default: ["text"]
- Description: Output types to generate. Use ["text", "audio"] for audio output.

### n
- Type: integer | null
- Default: 1
- Description: Number of completion choices to generate.

### parallel_tool_calls
- Type: boolean
- Default: true
- Description: Whether to enable parallel function calling during tool use.

### prediction
- Type: object
- Default: null
- Description: Configuration for Predicted Output to improve response times.
- Properties:
  - type (required): "content"
  - content (required): string | array of text parts — the predicted output content

### presence_penalty
- Type: number | null
- Default: 0
- Description: Number between -2.0 and 2.0. Positive values encourage new topics.

### prompt_cache_key
- Type: string
- Default: null
- Description: Used to cache responses for similar requests.

### prompt_cache_retention
- Type: string
- Default: null
- Description: Retention policy for prompt cache. Set to "24h" for extended caching.

### reasoning_effort
- Type: string
- Default: "medium"
- Description: Constrains reasoning effort for reasoning models.
- Values: none, minimal, low, medium, high, xhigh

### response_format
- Type: object
- Default: null
- Description: Format specification for structured outputs.
- Possible types (union):
  - Text Format:
    - type (required): "text"
  - JSON Object Format:
    - type (required): "json_object"
  - JSON Schema Format:
    - type (required): "json_schema"
    - json_schema (required): object
      - name (required): string — schema name
      - schema (optional): object — the JSON schema
      - description (optional): string
      - strict (optional): boolean — enable strict schema adherence

### safety_identifier
- Type: string
- Default: null
- Description: Stable identifier to detect policy violations. Recommended to hash user identifiers.

### service_tier
- Type: string
- Default: "auto"
- Description: Processing type.
- Values: auto, default, flex, priority

### stop
- Type: string | array | null
- Default: null
- Description: Up to 4 sequences where generation stops. Not supported with o3/o4-mini.

### store
- Type: boolean | null
- Default: false
- Description: Whether to store the completion for distillation/evals.

### stream
- Type: boolean | null
- Default: false
- Description: Enable streaming via server-sent events.

### stream_options
- Type: object
- Default: null
- Description: Options for streaming. Only set when stream: true.
- Properties:
  - include_usage (optional): boolean — include usage statistics in stream

### temperature
- Type: number
- Default: 1
- Description: Sampling temperature between 0 and 2. Higher = more random.

### tool_choice
- Type: string | object
- Default: auto (when tools present), none (otherwise)
- Description: Controls tool calling.
- Possible values:
  - "none" — model will not call any tool
  - "auto" — model can pick between generating a message or calling tools
  - "required" — model must call one or more tools
  - Named Tool Choice (object):
    - type (required): "function"
    - function (required): object with name (string)
  - Named Custom Tool Choice (object):
    - type (required): "custom"
    - custom (required): object with name (string)

### tools
- Type: array
- Default: null
- Description: List of tools the model may call.
- Possible types (union):
  - Function Tool:
    - type (required): "function"
    - function (required): object
      - name (required): string — function name (a-z, A-Z, 0-9, underscores)
      - description (optional): string — describes when to call
      - parameters (optional): object — JSON Schema for arguments
      - strict (optional): boolean — enable strict schema adherence
  - Custom Tool:
    - type (required): "custom"
    - custom (required): object
      - name (required): string — tool name
      - description (optional): string
      - format (optional): object — input format (text or grammar)

### top_logprobs
- Type: integer
- Default: null
- Description: Number of most likely tokens to return (0-20) at each position.

### top_p
- Type: number
- Default: 1
- Description: Nucleus sampling threshold. 0.1 = only top 10% probability mass.

### verbosity
- Type: string
- Default: "medium"
- Description: Response verbosity.
- Values: low, medium, high

### web_search_options
- Type: object
- Default: null
- Description: Configuration for web search tool.
- Properties:
  - search_context_size (optional): string — amount of context ("low" | "medium" | "high")
  - user_location (optional): object
    - type: "approximate"
    - approximate (optional): object with city, country, region, timezone

## Response Schema

### Chat Completion Object

#### id
- Type: string
- Description: Unique identifier for the chat completion.

#### object
- Type: string
- Description: Always "chat.completion".

#### created
- Type: integer
- Description: Unix timestamp (seconds) when the completion was created.

#### model
- Type: string
- Description: The model used for the completion.

#### choices
- Type: array
- Description: List of completion choices. Length depends on n parameter.

#### usage
- Type: object
- Description: Token usage statistics.

#### service_tier
- Type: string
- Description: Processing type used: default, flex, priority.

#### system_fingerprint
- Type: string
- Deprecated: Backend configuration fingerprint for determinism tracking.

### Choice Object

#### index
- Type: integer
- Description: Index of this choice in the choices array.

#### message
- Type: object
- Description: The generated message (see Message Object Response section).

#### finish_reason
- Type: string
- Description: Why generation stopped.
- Values:
  - stop — Natural stop or hit a stop sequence
  - length — Maximum token limit reached
  - tool_calls — Model invoked one or more tools
  - content_filter — Content was filtered due to policy
  - function_call — (Deprecated) Model invoked a function

#### logprobs
- Type: object | null
- Description: Log probability information (if requested).
- Properties:
  - content (array | null): array of token logprob objects
    - token (string): the token
    - logprob (number): log probability
    - bytes (array | null): UTF-8 byte representation
    - top_logprobs (array): top alternative tokens with logprobs

### Message Object (Response)

#### role
- Type: string
- Description: Always "assistant".

#### content
- Type: string | null
- Description: The text content of the response.

#### refusal
- Type: string | null
- Description: Refusal message if the model declined to respond.

#### audio
- Type: object | null
- Description: Audio response data when audio output is requested.
- Properties:
  - id (string): unique identifier for the audio
  - data (string): base64-encoded audio bytes
  - expires_at (integer): Unix timestamp when audio expires
  - transcript (string): transcript of the audio

#### tool_calls
- Type: array | null
- Description: Tool calls generated by the model.
- Item properties:
  - id (string): unique identifier for the tool call
  - type (string): "function"
  - function (object):
    - name (string): name of the function to call
    - arguments (string): JSON string of arguments (must be parsed)

#### function_call
- Type: object | null
- Deprecated: Use tool_calls instead.

#### annotations
- Type: array
- Description: Annotations for web search citations.
- Item properties:
  - type (string): "url_citation"
  - url_citation (object): with url, title, start_index, end_index

### Usage Object

#### prompt_tokens
- Type: integer
- Description: Tokens in the prompt.

#### completion_tokens
- Type: integer
- Description: Tokens in the generated completion.

#### total_tokens
- Type: integer
- Description: Total tokens used (prompt + completion).

#### prompt_tokens_details
- Type: object
- Description: Breakdown of prompt tokens.
  - cached_tokens (integer): Tokens served from cache
  - audio_tokens (integer): Tokens from audio input

#### completion_tokens_details
- Type: object
- Description: Breakdown of completion tokens.
  - reasoning_tokens (integer): Tokens used for reasoning (o-series models)
  - audio_tokens (integer): Tokens for audio output
  - accepted_prediction_tokens (integer): Accepted tokens from predicted output
  - rejected_prediction_tokens (integer): Rejected tokens from predicted output

---

## Streaming Response

When stream: true, the response is delivered as server-sent events. Each event contains a chat.completion.chunk object with a delta field containing incremental content.

### Chunk Object

#### id
- Type: string
- Description: Completion identifier.

#### object
- Type: string
- Description: Always "chat.completion.chunk".

#### created
- Type: integer
- Description: Unix timestamp.

#### model
- Type: string
- Description: Model used.

#### choices
- Type: array
- Description: Array with delta objects containing incremental content.

---

## Tool Calling

When using tools, the response may include tool_calls in the message.

### Tool Call Object (Response)

#### id
- Type: string
- Description: Unique identifier for the tool call.

#### type
- Type: string
- Description: The type of tool. Currently "function" for function tools.

#### function
- Type: object
- Description: The function that was called.
- Properties:
  - name (string): Name of the function
  - arguments (string): JSON string of arguments (must be parsed by caller)

### Function Tool Definition (Request)

When defining function tools in the tools array:

#### type
- Type: string
- Description: Must be "function".

#### function
- Type: object
- Properties:
  - name (required, string): Function name (a-z, A-Z, 0-9, underscores, max 64 chars)
  - description (optional, string): Description of what the function does
  - parameters (optional, object): JSON Schema defining the function's parameters
    - type: "object"
    - properties: object mapping parameter names to schemas
    - required: array of required parameter names
  - strict (optional, boolean): Enable strict schema adherence for Structured Outputs

---

## Examples

### Default

- Request

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "developer",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!"
    }
  ]
}
```

- Response

```json
{
  "id": "chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT",
  "object": "chat.completion",
  "created": 1741569952,
  "model": "gpt-4o-2024-08-06",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?",
        "refusal": null,
        "annotations": []
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 10,
    "total_tokens": 29,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "audio_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  },
  "service_tier": "default"
}
```

### Example Tool Call Response

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"Paris\"}"
      }
    }
  ]
}
```
