---
title: Batch
description: Process multiple requests asynchronously at lower cost
---

:::caution[Experimental API]
The Batch API is experimental and may change in future releases. Provider support is limited -- check the [providers page](/any-llm/providers/) for availability.
:::

The Batch API lets you submit multiple requests as a single job for asynchronous processing, typically at lower cost than real-time requests.

## How It Works

1. Prepare a JSONL file where each line is a batch request object.
2. Call `create_batch()` with the file path and target endpoint.
3. any-llm uploads the file to the provider and creates the batch job.
4. Poll with `retrieve_batch()` to check status.
5. When complete, download results from the provider.

### Input File Format

The input file must be a JSONL file where each line follows this structure:

```json
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Hello"}]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "World"}]}}
```

## `any_llm.create_batch()`

Create a batch job by uploading a local JSONL file.

```python
def create_batch(
    provider: str | LLMProvider,
    input_file_path: str,
    endpoint: str,
    *,
    completion_window: str = "24h",
    metadata: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Batch
```

### `any_llm.acreate_batch()`

Async variant with the same parameters.

## Parameters (create)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str \| LLMProvider` | *required* | Provider name (e.g., `"openai"`). |
| `input_file_path` | `str` | *required* | Path to a local JSONL file containing batch requests. |
| `endpoint` | `str` | *required* | Target endpoint for all requests (e.g., `"/v1/chat/completions"`). |
| `completion_window` | `str` | `"24h"` | Time frame within which the batch should be processed. |
| `metadata` | `dict[str, str] \| None` | `None` | Custom metadata for the batch. |
| `api_key` | `str \| None` | `None` | API key. Falls back to the provider's environment variable. |
| `api_base` | `str \| None` | `None` | Override the provider's base URL. |
| `client_args` | `dict[str, Any] \| None` | `None` | Provider-specific arguments for client initialization. |
| `**kwargs` | `Any` | | Additional provider-specific arguments. |

**Returns:** A [`Batch`](/any-llm/api/types/batch/) object.

## `any_llm.retrieve_batch()`

Retrieve the current status and details of a batch job.

```python
def retrieve_batch(
    provider: str | LLMProvider,
    batch_id: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Batch
```

### `any_llm.aretrieve_batch()`

Async variant with the same parameters.

## Parameters (retrieve)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str \| LLMProvider` | *required* | Provider name. |
| `batch_id` | `str` | *required* | The ID of the batch to retrieve. |
| `api_key` | `str \| None` | `None` | API key. |
| `api_base` | `str \| None` | `None` | Override base URL. |
| `client_args` | `dict[str, Any] \| None` | `None` | Provider-specific client arguments. |
| `**kwargs` | `Any` | | Additional provider-specific arguments. |

**Returns:** A [`Batch`](/any-llm/api/types/batch/) object.

## `any_llm.cancel_batch()`

Cancel an in-progress batch job.

```python
def cancel_batch(
    provider: str | LLMProvider,
    batch_id: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Batch
```

### `any_llm.acancel_batch()`

Async variant with the same parameters.

**Returns:** The cancelled [`Batch`](/any-llm/api/types/batch/) object.

## `any_llm.list_batches()`

List batch jobs for a provider.

```python
def list_batches(
    provider: str | LLMProvider,
    *,
    after: str | None = None,
    limit: int | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Sequence[Batch]
```

### `any_llm.alist_batches()`

Async variant with the same parameters.

## Parameters (list)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str \| LLMProvider` | *required* | Provider name. |
| `after` | `str \| None` | `None` | Pagination cursor. Returns batches created after this batch ID. |
| `limit` | `int \| None` | `None` | Maximum number of batches to return (default: 20). |
| `api_key` | `str \| None` | `None` | API key. |
| `api_base` | `str \| None` | `None` | Override base URL. |
| `client_args` | `dict[str, Any] \| None` | `None` | Provider-specific client arguments. |
| `**kwargs` | `Any` | | Additional provider-specific arguments. |

**Returns:** A `Sequence` of [`Batch`](/any-llm/api/types/batch/) objects.

## Usage

```python
from any_llm import create_batch, retrieve_batch, list_batches

# Create a batch job
batch = create_batch(
    provider="openai",
    input_file_path="requests.jsonl",
    endpoint="/v1/chat/completions",
)
print(f"Batch created: {batch.id}, status: {batch.status}")

# Check status
batch = retrieve_batch("openai", batch.id)
print(f"Status: {batch.status}")

# List all batches
batches = list_batches("openai")
for b in batches:
    print(f"{b.id}: {b.status}")
```
