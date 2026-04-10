---
title: Batch Types
description: Data models for batch operations
---

The `Batch` type represents a batch job returned by the [Batch API](../batch.md) functions.

## `Batch`

Re-exported from `openai.types.Batch`.

**Import:** `from any_llm.types.batch import Batch`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique batch identifier. |
| `object` | `str` | Always `"batch"`. |
| `endpoint` | `str` | The API endpoint used for all requests in the batch. |
| `input_file_id` | `str` | ID of the uploaded input file. |
| `completion_window` | `str` | Time frame for batch processing (e.g., `"24h"`). |
| `status` | `str` | Current status: `"validating"`, `"in_progress"`, `"finalizing"`, `"completed"`, `"failed"`, `"expired"`, `"cancelling"`, or `"cancelled"`. |
| `output_file_id` | `str \| None` | ID of the output file (available when status is `"completed"`). |
| `error_file_id` | `str \| None` | ID of the error file (if any requests failed). |
| `created_at` | `int` | Unix timestamp of batch creation. |
| `in_progress_at` | `int \| None` | Unix timestamp of when processing started. |
| `expires_at` | `int \| None` | Unix timestamp of when the batch expires. |
| `finalizing_at` | `int \| None` | Unix timestamp of when finalization started. |
| `completed_at` | `int \| None` | Unix timestamp of completion. |
| `failed_at` | `int \| None` | Unix timestamp of failure. |
| `expired_at` | `int \| None` | Unix timestamp of expiration. |
| `cancelling_at` | `int \| None` | Unix timestamp of cancellation request. |
| `cancelled_at` | `int \| None` | Unix timestamp of cancellation completion. |
| `request_counts` | `BatchRequestCounts \| None` | Counts of total, completed, and failed requests. |
| `metadata` | `dict[str, str] \| None` | Custom metadata attached to the batch. |

## `BatchRequestCounts`

Re-exported from `openai.types.batch_request_counts.BatchRequestCounts`.

**Import:** `from any_llm.types.batch import BatchRequestCounts`

| Field | Type | Description |
|-------|------|-------------|
| `total` | `int` | Total number of requests in the batch. |
| `completed` | `int` | Number of completed requests. |
| `failed` | `int` | Number of failed requests. |

## Usage

```python
from any_llm import create_batch, retrieve_batch

batch = create_batch(
    provider="openai",
    input_file_path="requests.jsonl",
    endpoint="/v1/chat/completions",
)

print(f"Batch ID: {batch.id}")
print(f"Status: {batch.status}")

# Poll for completion
import time
while batch.status not in ("completed", "failed", "expired", "cancelled"):
    time.sleep(30)
    batch = retrieve_batch("openai", batch.id)
    print(f"Status: {batch.status}")
    if batch.request_counts:
        print(f"  Completed: {batch.request_counts.completed}/{batch.request_counts.total}")

if batch.status == "completed":
    print(f"Output file: {batch.output_file_id}")
```

{% hint style="info" %}
The `Batch` and `BatchRequestCounts` types are direct re-exports from the OpenAI SDK. any-llm normalizes all provider batch responses into this format.
{% endhint %}
