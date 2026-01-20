# Acceptance Test Server

A FastAPI-based mock server for validating any-llm client implementations across multiple languages.

## Overview

This server acts as a **mock LLM provider** that validates incoming requests match expected schemas and scenarios. It helps ensure that different language implementations (Python, TypeScript, etc.) produce consistent, correct requests.

Results are stored in **SQLite** for persistence and can be grouped by **test run ID** for organized testing.

## Test Scenarios

Each scenario validates a specific capability:

| Scenario ID | Description | Key Validations |
|------------|-------------|-----------------|
| `basic_completion` | Simple chat completion | Required fields (model, messages), message format |
| `tool_calls` | Function/tool calling | Tools array format, tool_choice handling |
| `tool_response` | Tool response handling | Tool message format, tool_call_id matching |
| `streaming` | Streaming completions | stream=true, stream_options |
| `structured_output` | JSON schema response | response_format with json_schema |
| `multi_turn` | Multi-turn conversation | Message history format, role ordering |
| `system_message` | System prompt handling | System message position and format |
| `image_content` | Vision/image input | Image URL or base64 content parts |
| `temperature_params` | Generation parameters | temperature, top_p, max_tokens validation |

## Usage

### Starting the Server

```bash
uv run server.py
```

### Running Tests from Clients

Point your any-llm client to the acceptance test server and pass a test run ID:

**Python:**
```python
from any_llm import AnyLLM

llm = AnyLLM.create("openai", api_base="http://localhost:8080/v1")

# Run a scenario with a test run ID (via extra headers)
response = await llm.acompletion(
    model="test-basic",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={"X-Test-Run-Id": "python-v1.0.0"}
)
```

**TypeScript:**
```typescript
const llm = AnyLLM.create("openai", { apiBase: "http://localhost:8080/v1" });

const response = await llm.completion({
    model: "test-basic",
    messages: [{ role: "user", content: "Hello" }]
}, {
    headers: { "X-Test-Run-Id": "typescript-v1.0.0" }
});
```

## Model Naming Convention

The `model` parameter determines which scenario to validate:

| Model Name | Scenario |
|-----------|----------|
| `test-basic` | `basic_completion` |
| `test-tools` | `tool_calls` |
| `test-tool-response` | `tool_response` |
| `test-stream` | `streaming` |
| `test-structured` | `structured_output` |
| `test-multi-turn` | `multi_turn` |
| `test-system` | `system_message` |
| `test-image` | `image_content` |
| `test-params` | `temperature_params` |

## Response Format

The server returns OpenAI-compatible responses with validation metadata:

```json
{
    "id": "chatcmpl-test-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "test-basic",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Validation passed for scenario: basic_completion"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15
    },
    "_validation": {
        "scenario": "basic_completion",
        "passed": true,
        "errors": []
    }
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/scenarios` | GET | List available test scenarios |
| `/v1/models` | GET | List test models (OpenAI-compatible) |
| `/v1/chat/completions` | POST | Chat completion with validation |
| `/v1/test-runs` | GET | List all test runs |
| `/v1/test-runs` | POST | Create a new test run |
| `/v1/test-runs/{id}` | GET | Get a specific test run |
| `/v1/test-runs/{id}` | DELETE | Delete a test run and results |
| `/v1/test-runs/{id}/results` | GET | Get results for a test run |
| `/v1/test-runs/{id}/summary` | GET | Get summary for a test run |
| `/v1/results` | GET | Query all results (with filters) |
| `/v1/results` | DELETE | Clear all results |
| `/v1/summary` | GET | Get aggregated summary |
