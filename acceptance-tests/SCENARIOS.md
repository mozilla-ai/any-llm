# Shared Test Scenarios

This directory contains shared test scenario definitions used across all language implementations.

## Overview

To avoid duplication and ensure consistency across TypeScript, Python, and future language implementations, the acceptance test scenarios are defined once in a language-agnostic JSON format and served via the acceptance test server's API.

## Architecture

The test scenarios are:
1. **Defined** in `test-scenarios.json` as the canonical source
2. **Served** by the acceptance test server via the `/v1/test-data` endpoint
3. **Consumed** by language-specific test clients via HTTP requests

This ensures the server is the single source of truth and scenarios can be updated without redeploying test clients.

## Files

### `test-scenarios.json`

Contains the canonical test scenario definitions. Each scenario includes:
- `model`: The test model name to use
- `messages`: The conversation messages
- `stream`: (optional) Whether to use streaming
- `options`: (optional) Additional parameters like tools, response_format, temperature, etc.

### Server Endpoint

The acceptance test server exposes scenarios at:
```
GET http://localhost:8080/v1/test-data
```

Returns:
```json
{
  "scenarios": {
    "basic_completion": { ... },
    "tool_calls": { ... },
    ...
  }
}
```

### Usage

#### TypeScript (`ts/tests/acceptance.test.ts`)

```typescript
async function loadScenarios(): Promise<Record<string, any>> {
  const serverBase = BASE_URL.replace("/v1", "");
  const url = `${serverBase}/v1/test-data`;
  
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load test scenarios: ${response.status}`);
  }
  
  const data = await response.json();
  return data.scenarios;
}

const scenarios = await loadScenarios();
```

#### Python (`py/tests/test_acceptance.py`)

```python
async def load_scenarios() -> dict[str, Any]:
    """Load test scenarios from the acceptance test server API."""
    server_base = BASE_URL.replace("/v1", "")
    url = f"{server_base}/v1/test-data"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to load test scenarios: {response.status_code}")
        data = response.json()
        return data["scenarios"]

scenarios = await load_scenarios()
```

## Benefits

1. **Single Source of Truth**: Server provides canonical test data via API
2. **Easy Maintenance**: Update `test-scenarios.json` and restart server
3. **Language Agnostic**: JSON over HTTP works for all programming languages
4. **No File Dependencies**: Tests don't need access to scenario files
5. **Dynamic Updates**: Can modify scenarios without redeploying test clients
6. **Version Control**: Scenario changes tracked independently from test code
7. **Consistent State**: All test runs use the same scenario definitions

## Adding New Scenarios

1. Edit `test-scenarios.json`
2. Restart the acceptance test server
3. All language tests automatically use the new scenarios
4. No code changes needed in individual test files

## Scenario Naming Convention

Scenario keys use snake_case to be consistent across languages:
- `basic_completion`
- `tool_calls`
- `tool_response`
- `streaming`
- `structured_output`
- `multi_turn`
- `system_message`
- `image_content`
- `temperature_params`

## Parameter Naming

Note that parameter names in the JSON use snake_case (the OpenAI API convention), but each language implementation may need to transform these to match language-specific conventions:

- Python: Keep as-is (`stream_options`, `max_tokens`)
- TypeScript: Transform to camelCase (`streamOptions`, `maxTokens`)

Each language's test file handles this transformation appropriately.
