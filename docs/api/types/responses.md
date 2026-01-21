## OpenResponses Types

Data models and types for the [OpenResponses](https://www.openresponses.org/) API specification.

OpenResponses is an open-source specification for building multi-provider, interoperable LLM interfaces. These types support both standard OpenAI Responses API parameters and OpenResponses extensions.

### Core Types

| Type | Description |
|------|-------------|
| `Response` | OpenAI-compatible response object |
| `ResponseStreamEvent` | Streaming event from the API |
| `ResponsesParams` | Request parameters for the Responses API |

### Reasoning Types

| Type | Description |
|------|-------------|
| `ReasoningEffort` | Enum for reasoning depth (`none`, `low`, `medium`, `high`, `xhigh`) |
| `OpenResponsesReasoningConfig` | Configuration for reasoning behavior |
| `OpenResponsesReasoningItem` | Reasoning output with `content`, `encrypted_content`, `summary` |

### Tool Types

| Type | Description |
|------|-------------|
| `FunctionTool` | Standard function tool definition |
| `MCPTool` | [Model Context Protocol](https://modelcontextprotocol.io/) tool definition |
| `MCPToolApproval` | Approval mode for MCP tool execution |

### Streaming Types

| Type | Description |
|------|-------------|
| `OpenResponsesStreamEventType` | Enum of semantic event types |
| `OpenResponsesStreamEvent` | Base class for streaming events |
| `ReasoningDeltaEvent` | Streaming event for reasoning deltas |
| `ReasoningSummaryDeltaEvent` | Streaming event for reasoning summary deltas |

### Item Status

| Type | Description |
|------|-------------|
| `ItemStatus` | Lifecycle status (`in_progress`, `completed`, `failed`, `incomplete`) |

!!! info "Learn More"

    - [OpenResponses Specification](https://www.openresponses.org/specification)
    - [OpenResponses Reference](https://www.openresponses.org/reference)

::: any_llm.types.responses
