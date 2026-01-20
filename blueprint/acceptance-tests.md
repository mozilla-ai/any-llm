# Acceptance Test Client Generation Instructions

Generate a test file that validates an any-llm client implementation against the acceptance test server. The tests must cover all scenarios for all supported providers.

## Server Overview

The acceptance test server is a FastAPI mock LLM provider running at `http://localhost:8080` that validates incoming requests match OpenAI-compatible schemas. Results are stored per test run and can be queried for pass/fail status.

## Test Structure Requirements

### Test Run Setup

Before running tests, create a test run to group results:

```
POST /v1/test-runs?test_run_id={run_id}&description={description}
```

Include the test run ID in all subsequent requests via the `X-Test-Run-Id` header.

### Request Format

All requests go to `POST /v1/chat/completions` with OpenAI-compatible JSON body:

```json
{
  "model": "test-{scenario}",
  "messages": [...],
  // scenario-specific fields
}
```

### Response Validation

Non-streaming responses include `_validation` object:

```json
{
  "choices": [...],
  "_validation": {
    "passed": true|false,
    "errors": [{"field": "...", "message": "..."}],
    "scenario": "..."
  }
}
```

Streaming responses return validation status in headers:
- `X-Validation-Passed`: "true" or "false"
- `X-Scenario`: scenario ID

### Results Verification

After all tests, fetch the summary:
```
GET /v1/test-runs/{run_id}/summary
```

Response:
```json
{
  "total": 9,
  "passed": 9,
  "failed": 0,
  "by_scenario": {"basic_completion": {"passed": 1, "failed": 0}, ...}
}
```

## Required Test Payloads

### basic_completion
```json
{"model": "test-basic", "messages": [{"role": "user", "content": "Hello"}]}
```

### tool_calls
```json
{
  "model": "test-tools",
  "messages": [{"role": "user", "content": "What's the weather?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather for a location",
      "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
    }
  }]
}
```

### tool_response
```json
{
  "model": "test-tool-response",
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}}]},
    {"role": "tool", "content": "{\"temperature\": \"15C\"}", "tool_call_id": "call_123"}
  ]
}
```

### streaming
```json
{
  "model": "test-stream",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

### structured_output
```json
{
  "model": "test-structured",
  "messages": [{"role": "user", "content": "Give me a person"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    }
  }
}
```

### multi_turn
```json
{
  "model": "test-multi-turn",
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
  ]
}
```

### system_message
```json
{
  "model": "test-system",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
}
```

### image_content
```json
{
  "model": "test-image",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's in this image?"},
      {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ]
  }]
}
```

### temperature_params
```json
{
  "model": "test-params",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 100,
  "presence_penalty": 0.5,
  "frequency_penalty": 0.5
}
```

## Provider Testing

For each provider supported by the client library, configure the client to point to `http://localhost:8080/v1` as the API base and use a dummy API key. Run all 9 scenarios per provider to validate the client correctly formats requests regardless of provider configuration.

## Test File Structure

```
1. Setup: Create test run with unique ID (e.g., "{language}-{provider}-{timestamp}")
2. For each provider:
   a. Initialize client with provider config pointing to localhost:8080
   b. Run all 9 scenario tests
   c. Assert _validation.passed == true for each (or X-Validation-Passed header for streaming)
3. Teardown: Fetch summary, assert all tests passed, optionally clean up test run
```

## Assertions

For each test:
1. Request completes without HTTP error (status 200)
2. Response `_validation.passed` is `true` (or header `X-Validation-Passed: true` for streaming)
3. Response structure matches OpenAI format (has `choices`, `id`, `model`)

Final assertion: Summary shows `failed == 0` for the test run.
