---
title: Responses
description: OpenResponses API for agentic AI systems
---

The `responses` and `aresponses` functions implement the [OpenResponses specification](https://github.com/openresponsesspec/openresponses), a vendor-neutral API for agentic AI systems. This API supports multi-turn conversations, tool use, and streaming events.

## Return Types

The return type depends on the provider and whether streaming is enabled:

| Condition | Return Type |
|-----------|-------------|
| OpenResponses-compliant provider (non-streaming) | `openresponses_types.ResponseResource` |
| OpenAI-native provider (non-streaming) | `openai.types.responses.Response` |
| Streaming (`stream=True`) | `Iterator[ResponseStreamEvent]` (sync) or `AsyncIterator[ResponseStreamEvent]` (async) |

## `any_llm.responses()`

```
def responses(
    model: str,
    input_data: str | list[EasyInputMessageParam | Message | ResponseOutputMessageParam | ResponseFileSearchToolCallParam | ResponseComputerToolCallParam | ComputerCallOutput | ResponseFunctionWebSearchParam | ResponseFunctionToolCallParam | FunctionCallOutput | ToolSearchCall | ResponseToolSearchOutputItemParamParam | ResponseReasoningItemParam | ResponseCompactionItemParamParam | ImageGenerationCall | ResponseCodeInterpreterToolCallParam | LocalShellCall | LocalShellCallOutput | ShellCall | ShellCallOutput | ApplyPatchCall | ApplyPatchCallOutput | McpListTools | McpApprovalRequest | McpApprovalResponse | McpCall | ResponseCustomToolCallOutputParam | ResponseCustomToolCallParam | ItemReference],
    *,
    provider: str | LLMProvider | None = None,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    max_tool_calls: int | None = None,
    parallel_tool_calls: bool | None = None,
    reasoning: Any | None = None,
    text: Any | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    truncation: str | None = None,
    store: bool | None = None,
    service_tier: str | None = None,
    user: str | None = None,
    metadata: dict[str, str] | None = None,
    previous_response_id: str | None = None,
    include: list[str] | None = None,
    background: bool | None = None,
    safety_identifier: str | None = None,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: str | None = None,
    conversation: str | dict[str, Any] | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ResponseResource | Response | Iterator[ResponseAudioDeltaEvent | ResponseAudioDoneEvent | ResponseAudioTranscriptDeltaEvent | ResponseAudioTranscriptDoneEvent | ResponseCodeInterpreterCallCodeDeltaEvent | ResponseCodeInterpreterCallCodeDoneEvent | ResponseCodeInterpreterCallCompletedEvent | ResponseCodeInterpreterCallInProgressEvent | ResponseCodeInterpreterCallInterpretingEvent | ResponseCompletedEvent | ResponseContentPartAddedEvent | ResponseContentPartDoneEvent | ResponseCreatedEvent | ResponseErrorEvent | ResponseFileSearchCallCompletedEvent | ResponseFileSearchCallInProgressEvent | ResponseFileSearchCallSearchingEvent | ResponseFunctionCallArgumentsDeltaEvent | ResponseFunctionCallArgumentsDoneEvent | ResponseInProgressEvent | ResponseFailedEvent | ResponseIncompleteEvent | ResponseOutputItemAddedEvent | ResponseOutputItemDoneEvent | ResponseReasoningSummaryPartAddedEvent | ResponseReasoningSummaryPartDoneEvent | ResponseReasoningSummaryTextDeltaEvent | ResponseReasoningSummaryTextDoneEvent | ResponseReasoningTextDeltaEvent | ResponseReasoningTextDoneEvent | ResponseRefusalDeltaEvent | ResponseRefusalDoneEvent | ResponseTextDeltaEvent | ResponseTextDoneEvent | ResponseWebSearchCallCompletedEvent | ResponseWebSearchCallInProgressEvent | ResponseWebSearchCallSearchingEvent | ResponseImageGenCallCompletedEvent | ResponseImageGenCallGeneratingEvent | ResponseImageGenCallInProgressEvent | ResponseImageGenCallPartialImageEvent | ResponseMcpCallArgumentsDeltaEvent | ResponseMcpCallArgumentsDoneEvent | ResponseMcpCallCompletedEvent | ResponseMcpCallFailedEvent | ResponseMcpCallInProgressEvent | ResponseMcpListToolsCompletedEvent | ResponseMcpListToolsFailedEvent | ResponseMcpListToolsInProgressEvent | ResponseOutputTextAnnotationAddedEvent | ResponseQueuedEvent | ResponseCustomToolCallInputDeltaEvent | ResponseCustomToolCallInputDoneEvent]
```

## `any_llm.aresponses()`

Async variant with the same parameters. Returns `ResponseResource | Response | AsyncIterator[ResponseStreamEvent]`.

```
async def aresponses(
    model: str,
    input_data: str | list[EasyInputMessageParam | Message | ResponseOutputMessageParam | ResponseFileSearchToolCallParam | ResponseComputerToolCallParam | ComputerCallOutput | ResponseFunctionWebSearchParam | ResponseFunctionToolCallParam | FunctionCallOutput | ToolSearchCall | ResponseToolSearchOutputItemParamParam | ResponseReasoningItemParam | ResponseCompactionItemParamParam | ImageGenerationCall | ResponseCodeInterpreterToolCallParam | LocalShellCall | LocalShellCallOutput | ShellCall | ShellCallOutput | ApplyPatchCall | ApplyPatchCallOutput | McpListTools | McpApprovalRequest | McpApprovalResponse | McpCall | ResponseCustomToolCallOutputParam | ResponseCustomToolCallParam | ItemReference],
    *,
    provider: str | LLMProvider | None = None,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    max_tool_calls: int | None = None,
    parallel_tool_calls: bool | None = None,
    reasoning: Any | None = None,
    text: Any | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    truncation: str | None = None,
    store: bool | None = None,
    service_tier: str | None = None,
    user: str | None = None,
    metadata: dict[str, str] | None = None,
    previous_response_id: str | None = None,
    include: list[str] | None = None,
    background: bool | None = None,
    safety_identifier: str | None = None,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: str | None = None,
    conversation: str | dict[str, Any] | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ResponseResource | Response | AsyncIterator[ResponseAudioDeltaEvent | ResponseAudioDoneEvent | ResponseAudioTranscriptDeltaEvent | ResponseAudioTranscriptDoneEvent | ResponseCodeInterpreterCallCodeDeltaEvent | ResponseCodeInterpreterCallCodeDoneEvent | ResponseCodeInterpreterCallCompletedEvent | ResponseCodeInterpreterCallInProgressEvent | ResponseCodeInterpreterCallInterpretingEvent | ResponseCompletedEvent | ResponseContentPartAddedEvent | ResponseContentPartDoneEvent | ResponseCreatedEvent | ResponseErrorEvent | ResponseFileSearchCallCompletedEvent | ResponseFileSearchCallInProgressEvent | ResponseFileSearchCallSearchingEvent | ResponseFunctionCallArgumentsDeltaEvent | ResponseFunctionCallArgumentsDoneEvent | ResponseInProgressEvent | ResponseFailedEvent | ResponseIncompleteEvent | ResponseOutputItemAddedEvent | ResponseOutputItemDoneEvent | ResponseReasoningSummaryPartAddedEvent | ResponseReasoningSummaryPartDoneEvent | ResponseReasoningSummaryTextDeltaEvent | ResponseReasoningSummaryTextDoneEvent | ResponseReasoningTextDeltaEvent | ResponseReasoningTextDoneEvent | ResponseRefusalDeltaEvent | ResponseRefusalDoneEvent | ResponseTextDeltaEvent | ResponseTextDoneEvent | ResponseWebSearchCallCompletedEvent | ResponseWebSearchCallInProgressEvent | ResponseWebSearchCallSearchingEvent | ResponseImageGenCallCompletedEvent | ResponseImageGenCallGeneratingEvent | ResponseImageGenCallInProgressEvent | ResponseImageGenCallPartialImageEvent | ResponseMcpCallArgumentsDeltaEvent | ResponseMcpCallArgumentsDoneEvent | ResponseMcpCallCompletedEvent | ResponseMcpCallFailedEvent | ResponseMcpCallInProgressEvent | ResponseMcpListToolsCompletedEvent | ResponseMcpListToolsFailedEvent | ResponseMcpListToolsInProgressEvent | ResponseOutputTextAnnotationAddedEvent | ResponseQueuedEvent | ResponseCustomToolCallInputDeltaEvent | ResponseCustomToolCallInputDoneEvent]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | Model identifier. **Recommended**: Use with separate `provider` parameter (e.g., model='gpt-4o', provider='openai'). **Alternative**: Combined format 'provider:model' (e.g., 'openai:gpt-4o'). Legacy format 'provider/model' is also supported but deprecated. |
| `input_data` | `str \| list[EasyInputMessageParam \| Message \| ResponseOutputMessageParam \| ResponseFileSearchToolCallParam \| ResponseComputerToolCallParam \| ComputerCallOutput \| ResponseFunctionWebSearchParam \| ResponseFunctionToolCallParam \| FunctionCallOutput \| ToolSearchCall \| ResponseToolSearchOutputItemParamParam \| ResponseReasoningItemParam \| ResponseCompactionItemParamParam \| ImageGenerationCall \| ResponseCodeInterpreterToolCallParam \| LocalShellCall \| LocalShellCallOutput \| ShellCall \| ShellCallOutput \| ApplyPatchCall \| ApplyPatchCallOutput \| McpListTools \| McpApprovalRequest \| McpApprovalResponse \| McpCall \| ResponseCustomToolCallOutputParam \| ResponseCustomToolCallParam \| ItemReference]` | *required* | The input payload accepted by provider's Responses API. For OpenAI-compatible providers, this is typically a list mixing text, images, and tool instructions, or a dict per OpenAI spec. |
| `provider` | `str \| LLMProvider \| None` | None | **Recommended**: Provider name to use for the request (e.g., 'openai', 'mistral'). When provided, the model parameter should contain only the model name. |
| `tools` | `list[dict[str, Any] \| Callable[..., Any]] \| None` | None | Optional tools for tool calling (Python callables or OpenAI tool dicts) |
| `tool_choice` | `str \| dict[str, Any] \| None` | None | Controls which tools the model can call |
| `max_output_tokens` | `int \| None` | None | Maximum number of output tokens to generate |
| `temperature` | `float \| None` | None | Controls randomness in the response (0.0 to 2.0) |
| `top_p` | `float \| None` | None | Controls diversity via nucleus sampling (0.0 to 1.0) |
| `stream` | `bool \| None` | None | Whether to stream response events |
| `api_key` | `str \| None` | None | API key for the provider |
| `api_base` | `str \| None` | None | Base URL for the provider API |
| `instructions` | `str \| None` | None | A system (or developer) message inserted into the model's context. |
| `max_tool_calls` | `int \| None` | None | The maximum number of total calls to built-in tools that can be processed in a response. This maximum number applies across all built-in tool calls, not per individual tool. Any further attempts to call a tool by the model will be ignored. |
| `parallel_tool_calls` | `bool \| None` | None | Whether to allow the model to run tool calls in parallel. |
| `reasoning` | `Any \| None` | None | Configuration options for reasoning models. |
| `text` | `Any \| None` | None | Configuration options for a text response from the model. Can be plain text or structured JSON data. |
| `presence_penalty` | `float \| None` | None | Penalizes new tokens based on whether they appear in the text so far. |
| `frequency_penalty` | `float \| None` | None | Penalizes new tokens based on their frequency in the text so far. |
| `truncation` | `str \| None` | None | Controls how the service truncates input when it exceeds the model context window. |
| `store` | `bool \| None` | None | Whether to store the response so it can be retrieved later. |
| `service_tier` | `str \| None` | None | The service tier to use for this request. |
| `user` | `str \| None` | None | A unique identifier representing your end user. |
| `metadata` | `dict[str, str] \| None` | None | Key-value pairs for custom metadata (up to 16 pairs). |
| `previous_response_id` | `str \| None` | None | The ID of the response to use as the prior turn for this request. |
| `include` | `list[str] \| None` | None | Items to include in the response (e.g., 'reasoning.encrypted_content'). |
| `background` | `bool \| None` | None | Whether to run the request in the background and return immediately. |
| `safety_identifier` | `str \| None` | None | A stable identifier used for safety monitoring and abuse detection. |
| `prompt_cache_key` | `str \| None` | None | A key to use when reading from or writing to the prompt cache. |
| `prompt_cache_retention` | `str \| None` | None | How long to retain a prompt cache entry created by this request. |
| `conversation` | `str \| dict[str, Any] \| None` | None | The conversation to associate this response with (ID string or ConversationParam object). |
| `client_args` | `dict[str, Any] \| None` | None | Additional provider-specific arguments that will be passed to the provider's client instantiation. |
| `**kwargs` | `Any` | *required* | Additional provider-specific arguments that will be passed to the provider's API call. |

## Usage

### Basic response

```python
from any_llm import responses

result = responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="What is the capital of France?",
)
print(result.output_text)
```

### With instructions

```python
result = responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="Translate to French: Hello, how are you?",
    instructions="You are a professional translator. Always respond with only the translation.",
)
```

### Streaming

```python
for event in responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="Tell me a short story.",
    stream=True,
):
    print(event)
```

### Multi-turn with `previous_response_id`

```python
first = responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="My name is Alice.",
    store=True,
)

second = responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="What is my name?",
    previous_response_id=first.id,
)
```

{% hint style="info" %}
Not all providers support the Responses API. Check the [providers page](../providers.md) for support details, or query `ProviderMetadata.responses` programmatically.
{% endhint %}
