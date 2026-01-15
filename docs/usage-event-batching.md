# Usage Event Batching

When using the `platform` provider with any-llm, usage events are automatically batched to reduce HTTP overhead and improve performance. This document explains how batching works and how to optimize it for your use case.

## Overview

By default, every LLM completion generates a usage event that tracks tokens, costs, and performance metrics. Without batching, each event would require a separate HTTP request to the platform API. With batching enabled (the default), events are collected and sent in groups, reducing network overhead by up to **98%**.

## How It Works

The batching system uses a queue that automatically flushes based on two triggers:

- **Batch size**: When the queue reaches a certain number of events (default: 50)
- **Time window**: After a certain time period has elapsed (default: 5 seconds)

Whichever trigger fires first will cause the queued events to be sent to the platform API.

### Automatic Cleanup

The batching system automatically flushes remaining events when your process exits, ensuring no data loss. This happens via Python's `atexit` handler, which runs on:

- Normal process exit
- Keyboard interrupt (Ctrl+C)
- Most graceful shutdowns

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

Batching is enabled by default with sensible defaults:

```python
# Example usage with the platform provider:
# from any_llm import completion
# 
# # Events are automatically batched with default settings:
# # - batch_size: 50 events
# # - flush_interval: 5.0 seconds
# response = completion(
#     provider="platform",
#     model="gpt-4",
#     messages=[{"role": "user", "content": "Hello!"}],
#     api_key="your-any-llm-platform-key"
# )
pass
```

### Custom Configuration

You can customize batching behavior if needed, though the defaults work well for most use cases:

```python
from any_llm.providers.platform import queue_completion_usage_event

# Example: Smaller batches, more frequent flushes
# async def custom_batching():
#     await queue_completion_usage_event(
#         platform_client=platform_client,
#         client=http_client,
#         any_llm_key="your-key",
#         provider="openai",
#         completion=response,
#         provider_key_id="key-id",
#         batch_size=25,           # Flush every 25 events (instead of 50)
#         flush_interval=2.0,      # Flush every 2 seconds (instead of 5)
#     )
```

## Manual Control

### Manual Flush

Force immediate sending of all queued events:

```python
from any_llm.providers.platform import shutdown_global_batch_queue

# Example: Flush all pending events immediately
# async def manual_flush():
#     await shutdown_global_batch_queue()
```

### When to Manual Flush

Manual flushing is rarely needed since automatic cleanup handles most cases, but consider it for:

- **End of batch jobs**: After processing a large batch of requests
- **Critical checkpoints**: Before important state transitions
- **Testing/debugging**: To see usage events immediately

## Best Practices

### For High-Throughput Applications

Use larger batch sizes to maximize efficiency:

```python
# Example configuration for high throughput:
# batch_size = 100        # Larger batches
# flush_interval = 10.0   # Longer wait time
pass  # Configuration would be passed to queue_completion_usage_event()
```

### For Real-Time Applications

Use smaller batches for lower latency:

```python
# Example configuration for real-time applications:
# batch_size = 10         # Smaller batches
# flush_interval = 1.0    # More frequent flushes
pass  # Configuration would be passed to queue_completion_usage_event()
```

### For Long-Running Services

While automatic cleanup handles graceful shutdowns, explicitly call shutdown for cleaner logs:

```python
# Example shutdown handler:
# import logging
# logger = logging.getLogger(__name__)
# 
# async def shutdown_handler():
#     """Graceful shutdown for long-running services."""
#     logger.info("Shutting down...")
#     
#     # Flush remaining usage events
#     await shutdown_global_batch_queue()
#     
#     # Other cleanup...
pass
```

## Monitoring

The batching system logs important events:

```
DEBUG: Registered automatic cleanup handler for usage event batching
DEBUG: Successfully sent batch of 50 usage events
INFO: Flushing remaining usage events on process exit...
INFO: Successfully flushed remaining usage events
```

## Migration from Individual Events

If you were previously using `post_completion_usage_event()` (now deprecated), simply replace it with `queue_completion_usage_event()`:

```python
# Old (deprecated) - sends immediately:
# await post_completion_usage_event(
#     platform_client=client,
#     client=http_client,
#     any_llm_key="key",
#     provider="openai",
#     completion=response,
#     provider_key_id="key-id"
# )

# New (recommended) - uses batching:
# await queue_completion_usage_event(
#     platform_client=client,
#     client=http_client,
#     any_llm_key="key",
#     provider="openai",
#     completion=response,
#     provider_key_id="key-id"
# )
pass
```

The `platform` provider automatically uses batching, so no code changes are needed if you're using the high-level `completion()` API.

## Technical Details

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

1. **Check flush timing**: Events may still be queued. Wait for flush interval or call `shutdown_global_batch_queue()`
2. **Verify API key**: Ensure your `ANY_LLM_KEY` is valid and has proper permissions
3. **Check logs**: Look for error messages about failed batch sends

### High Latency

If you're seeing delays in usage analytics:

- Reduce `flush_interval` for more frequent updates
- Reduce `batch_size` if you have low request volume
- Consider if 5-second delay is acceptable for your use case

### Memory Concerns

If batching uses too much memory:

- Reduce `batch_size` to limit queue size
- Reduce `flush_interval` to clear queue more frequently
- Monitor with `len(queue._queue)` if needed

## Related

- [Platform Provider Documentation](providers.md)
- [Performance Optimization Guide](docs/performance.md)
- [any-llm-gateway Overview](gateway/overview.md)
