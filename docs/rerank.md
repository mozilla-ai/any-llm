---
title: Rerank
description: Rerank documents by relevance to a query
---

## Overview

The rerank API lets you reorder a list of documents by their relevance to a query. This is useful for search pipelines, retrieval-augmented generation (RAG), and any workflow where you need to select the most relevant documents from a candidate set.

## Supported Providers

| Provider | `SUPPORTS_RERANK` | Notes |
|----------|-------------------|-------|
| Cohere | Yes | Native rerank via the Cohere V2 SDK |
| Gateway | Yes | Proxies to `/v1/rerank` on an upstream server |

All other providers return `SUPPORTS_RERANK = False` and raise `NotImplementedError` if called.

## Installation

Rerank with Cohere requires the Cohere provider extra:

```bash
pip install any-llm-sdk[cohere]
```

## Quick Start

### Synchronous

```python
from any_llm import rerank

response = rerank(
    "cohere:rerank-v3.5",
    query="What is machine learning?",
    documents=[
        "Machine learning is a subset of artificial intelligence.",
        "The weather today is sunny.",
        "Deep learning uses neural networks with many layers.",
    ],
    top_n=2,
)

for result in response.results:
    print(f"  index={result.index}  score={result.relevance_score:.4f}")
```

An async variant `arerank()` is also available with the same signature.

## API Reference

### `rerank()` / `arerank()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Provider-prefixed model ID (e.g. `"cohere:rerank-v3.5"`) |
| `query` | `str` | required | The search query string |
| `documents` | `list[str]` | required | Documents to rerank |
| `top_n` | `int \| None` | `None` | Maximum number of results to return. Defaults to all documents. |
| `max_tokens_per_doc` | `int \| None` | `None` | Per-document token truncation limit |
| `provider` | `str \| None` | `None` | Explicit provider name. Inferred from `model` if omitted. |
| `api_key` | `str \| None` | `None` | Provider API key. Falls back to the environment variable. |
| `api_base` | `str \| None` | `None` | Provider API base URL. Falls back to the environment variable. |
| `client_args` | `dict \| None` | `None` | Additional arguments for the provider client constructor |

### `RerankResponse`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str \| None` | Provider-assigned response ID (may be `None`) |
| `results` | `list[RerankResult]` | Results sorted by `relevance_score` descending |
| `meta` | `RerankMeta \| None` | Provider-specific billing metadata |
| `usage` | `RerankUsage \| None` | Normalized token usage |

### `RerankResult`

| Field | Type | Description |
|-------|------|-------------|
| `index` | `int` | Zero-based index into the original `documents` list |
| `relevance_score` | `float` | Relevance score (higher is more relevant) |

## Gateway Provider

The gateway provider (`provider="gateway"`) can proxy rerank requests to an upstream server that exposes a `/v1/rerank` endpoint. The gateway provider sends the request via HTTP with correct authentication headers. Note that the any-llm gateway server does not yet expose a `/v1/rerank` route itself.
