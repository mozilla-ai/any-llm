# Usage Event Batching

When using the `platform` provider with any-llm, usage events are automatically batched to reduce HTTP overhead and improve performance. This document explains how batching works and how to optimize it for your use case.

## Overview

By default, every LLM completion generates a usage event that tracks tokens, costs, and performance metrics. Without batching, each event would require a separate HTTP request to the platform API. With batching enabled (the default), events are collected and sent in groups, reducing network overhead by up to **98%**.

## How It Works

Each `PlatformProvider` instance maintains its own batching queue that automatically flushes based on two triggers:

- **Batch size**: When the queue reaches a certain number of events (default: 50)
- **Time window**: After a certain time period has elapsed (default: 5 seconds)

Whichever trigger fires first will cause the queued events to be sent to the platform API.

### Automatic Cleanup

The `PlatformProvider` supports async context managers for automatic cleanup:

```python
from any_llm.providers.platform import PlatformProvider
from any_llm.providers.openai import OpenaiProvider

async def example():
    async with PlatformProvider(api_key="your-key") as provider:
        provider.provider = OpenaiProvider
        # Use provider for completions
        # Events are automatically flushed when exiting the context
```

You can also manually flush or shutdown:

```python
async def example():
    provider = PlatformProvider(api_key="your-key")
    # ... use provider ...

    # Flush pending events without shutting down
    await provider.flush_usage_events()

    # Or shutdown completely (flushes and stops background tasks)
    await provider.shutdown()
```

## Performance Impact

Based on testing with various workloads:

| Events | Without Batching | With Batching | Reduction |
|--------|------------------|---------------|-----------|
| 50     | 50 requests      | 1 request     | 98%       |
| 100    | 100 requests     | 2 requests    | 98%       |
| 500    | 500 requests     | 10 requests   | 98%       |
| 1000   | 1000 requests    | 20 requests   | 98%       |

## Configuration

### Default Behavior

Batching is enabled by default when you create a `PlatformProvider`:

```python
from any_llm.providers.platform import PlatformProvider
from any_llm.providers.openai import OpenaiProvider

async def example():
    # Events are automatically batched with default settings:
    # - batch_size: 50 events
    # - flush_interval: 5.0 seconds
    provider = PlatformProvider(api_key="your-any-llm-platform-key")
    provider.provider = OpenaiProvider

    # Use the provider normally
    response = await provider.acompletion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

### Custom Configuration

You can customize batching behavior per provider instance:

```python
from any_llm.providers.platform import PlatformProvider

# Smaller batches, more frequent flushes
provider = PlatformProvider(
    api_key="your-key",
    batch_size=25,         # Flush every 25 events (instead of 50)
    flush_interval=2.0,    # Flush every 2 seconds (instead of 5)
)
```

## Manual Control

### Manual Flush

Force immediate sending of all queued events for a specific provider:

```python
from any_llm.providers.platform import PlatformProvider

async def example():
    provider = PlatformProvider(api_key="your-key")
    # ... use provider ...

    # Flush all pending events immediately
    await provider.flush_usage_events()
```

### Graceful Shutdown

For long-running applications, explicitly shutdown the provider:

```python
async def example():
    provider = PlatformProvider(api_key="your-key")
    try:
        # ... use provider ...
        pass
    finally:
        # Flush remaining events and stop background tasks
        await provider.shutdown()
```

### When to Manual Flush

Manual flushing is useful for:

- **End of batch jobs**: After processing a large batch of requests
- **Critical checkpoints**: Before important state transitions  
- **Testing/debugging**: To see usage events immediately
- **Graceful shutdown**: To ensure all events are sent before exit

## Best Practices

### For High-Throughput Applications

Use larger batch sizes to maximize efficiency:

```python
from any_llm.providers.platform import PlatformProvider

# Configuration for high throughput
provider = PlatformProvider(
    api_key="your-key",
    batch_size=100,        # Larger batches
    flush_interval=10.0,   # Longer wait time
)
```

### For Real-Time Applications

Use smaller batches for lower latency:

```python
from any_llm.providers.platform import PlatformProvider

# Configuration for real-time applications
provider = PlatformProvider(
    api_key="your-key",
    batch_size=10,         # Smaller batches
    flush_interval=1.0,    # More frequent flushes
)
```

### For Long-Running Services

Use async context managers or explicit shutdown for cleaner cleanup:

```python
from any_llm.providers.platform import PlatformProvider
from any_llm.providers.openai import OpenaiProvider

async def example():
    # Option 1: Context manager (recommended)
    async with PlatformProvider(api_key="your-key") as provider:
        provider.provider = OpenaiProvider
        # ... use provider ...
        # Automatically flushes on exit

    # Option 2: Manual shutdown
    provider = PlatformProvider(api_key="your-key")
    try:
        provider.provider = OpenaiProvider
        # ... use provider ...
    finally:
        await provider.shutdown()
```

## Monitoring

Each provider instance logs batching events:

```
DEBUG: Successfully sent batch of 50 usage events
```

You can also inspect the queue directly if needed:

```python
provider = PlatformProvider(api_key="your-key")
# Check queue size
queue_size = len(provider.batch_queue._queue)
```

## Migration from Individual Events

If you were previously using `post_completion_usage_event()` (now deprecated), the batching is now automatic when using `PlatformProvider`. No code changes needed - batching happens transparently.

If you need direct access to the batching queue (advanced usage):

```python
from any_llm.providers.platform import PlatformProvider

# The provider creates and manages its own queue
provider = PlatformProvider(api_key="your-key")
# Access the queue directly if needed
batch_queue = provider.batch_queue
```

Or create a standalone queue (advanced - requires any-llm-platform-client):

```text
from any_llm.providers.platform import UsageEventBatchQueue
from any_llm_platform_client import AnyLLMPlatformClient
import httpx

platform_client = AnyLLMPlatformClient(any_llm_platform_url="...")
http_client = httpx.AsyncClient()
queue = UsageEventBatchQueue(
    platform_client=platform_client,
    http_client=http_client,
    batch_size=50,
    flush_interval=5.0
)
```

## Technical Details

### Instance-Based Queues

Each `PlatformProvider` instance maintains its own `UsageEventBatchQueue`. This means:

- Different provider instances have independent queues
- Each queue authenticates with its own credentials
- No shared global state between providers
- Predictable lifecycle management

### Thread Safety

The batching queue uses asyncio locks to ensure thread-safe enqueueing and flushing operations. Multiple concurrent requests can safely add events to the queue.

### API Key Grouping

Events are automatically grouped by API key before sending. This ensures that events for different projects are sent with the correct authentication.

### Error Handling

If a batch fails to send:
- The error is logged but doesn't crash your application
- Other batches (e.g., for different API keys) continue normally
- Failed events are not automatically retried

### Memory Usage

With default settings (50 events per batch, 5-second flush):
- Maximum queue size: ~50 events
- Memory per event: ~1-2 KB
- Total memory overhead: < 100 KB

## Troubleshooting

### Events Not Appearing in Platform

1. **Check flush timing**: Events may still be queued. Manually flush with `await provider.flush_usage_events()`
2. **Verify API key**: Ensure your `ANY_LLM_KEY` is valid and has proper permissions
3. **Check logs**: Look for error messages about failed batch sends

### High Latency

If you're seeing delays in usage analytics:

- Reduce `flush_interval` when creating the provider for more frequent updates
- Reduce `batch_size` if you have low request volume
- Consider if 5-second delay is acceptable for your use case

### Memory Concerns

If batching uses too much memory:

- Reduce `batch_size` when creating the provider to limit queue size
- Reduce `flush_interval` to clear queue more frequently
- Monitor with `len(provider.batch_queue._queue)` if needed

## Related

- [Platform Provider Documentation](providers.md)
- [any-llm-gateway Overview](gateway/overview.md)
