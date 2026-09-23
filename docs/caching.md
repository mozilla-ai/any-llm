---
title: Context Caching
description: Create and manage provider-hosted context caches through an AnyLLM instance
---

# Context Caching

The context-cache API stores a reusable prompt prefix (system instructions, tools,
and conversation turns) on the provider, so later requests can reference it instead
of resending it. It exposes create, list, retrieve, update, and delete on an `AnyLLM`
instance. Gemini (Developer API) and Vertex AI support it; other providers raise
`NotImplementedError`.

| Provider | Create | List | Retrieve | Update | Delete |
| --- | --- | --- | --- | --- | --- |
| Gemini (Developer API) | Yes | Yes | Yes | Yes | Yes |
| Vertex AI | Yes | Yes | Yes | Yes | Yes |

Check `get_provider_metadata().cache_operations` before using caches on a provider.

```python
from any_llm import AnyLLM

provider = AnyLLM.create("gemini")  # GEMINI_API_KEY or GOOGLE_API_KEY
capabilities = provider.get_provider_metadata()
print(capabilities.caches)
print(capabilities.cache_operations)
```

## Create a cache and use it

`messages`, `tools`, and `tool_choice` take the same shapes as `completion`, and are
converted the same way a completion converts them. System and developer messages
become the cache's system instruction; the other messages become its contents.

```python
from any_llm import AnyLLM

provider = AnyLLM.create("gemini")
manual = "..."  # Your long, reused context, above the model's minimum cache size
cache = provider.create_cache(
    "gemini-2.5-flash",
    messages=[
        {"role": "system", "content": "Answer questions about the attached manual."},
        {"role": "user", "content": manual},
    ],
    ttl=3600,
    display_name="product-manual",
)
try:
    response = provider.completion(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "How do I reset the device?"}],
        cached_content=cache.id,
    )
    print(response.choices[0].message.content)
    print(cache.usage.total_tokens if cache.usage else None)
finally:
    provider.delete_cache(cache.id)
```

`cached_content` is passed through to Gemini's `GenerateContentConfig` like any other
provider-specific keyword, so it works with `completion`, `acompletion`, and streaming.
The completion must use the same model as the cache.

Gemini rejects a request that uses `cached_content` and also sends a system
instruction, tools, or a tool configuration: those live in the cache. When
completing against a cache, send only the new turns, and do not pass `system`
or `developer` messages, `tools`, or `tool_choice`. Put everything the model
needs for that part of the prompt into the cache when you create it.

Gemini enforces a minimum cached token count, which depends on the model; a
smaller cache is rejected with `InvalidRequestError` (HTTP 400).

## Expiry

`ttl` is a positive integer number of seconds from now; `expires_at` is a
timezone-aware `datetime`. They are mutually exclusive. On creation, omitting both
leaves the provider's default lifetime (one hour on Gemini). `update_cache` changes
only the expiry and requires exactly one of the two.

```python
from datetime import UTC, datetime, timedelta

provider.update_cache(cache.id, ttl=7200)
provider.update_cache(cache.id, expires_at=datetime.now(UTC) + timedelta(days=1))
```

The cached content, system instruction, and tools cannot be changed after creation;
create a new cache instead.

## Cache IDs

Cache IDs are the provider's resource names. Gemini returns `cachedContents/{id}`;
passing the bare `{id}` is accepted and canonicalized. Vertex AI returns
`projects/{project}/locations/{location}/cachedContents/{id}`, which is accepted as
is. IDs containing whitespace, URL delimiters, or dot segments raise
`InvalidRequestError` before any request is sent.

## Methods

| Synchronous | Asynchronous | Result |
| --- | --- | --- |
| `create_cache(model, messages=None, tools=None, tool_choice=None, ttl=None, expires_at=None, display_name=None, **kwargs)` | `await acreate_cache(...)` | `CachedContent` |
| `list_caches(limit=None, cursor=None, **kwargs)` | `await alist_caches(...)` | `CachePage` |
| `retrieve_cache(cache_id, **kwargs)` | `await aretrieve_cache(...)` | `CachedContent` |
| `update_cache(cache_id, ttl=None, expires_at=None, **kwargs)` | `await aupdate_cache(...)` | `CachedContent` |
| `delete_cache(cache_id, **kwargs)` | `await adelete_cache(...)` | `CacheDeleted` |

`model` is the model ID without the provider prefix, as it is on the instance's
other methods. These are instance methods: reuse the instance configured for the
originating provider account, since cache IDs are not portable across accounts or
providers. Sync methods honor the existing `allow_running_loop` option.

`CachedContent`, `CacheUsage`, `CachePage`, and `CacheDeleted` are exported from
`any_llm` and `any_llm.types.caches`. `CachedContent` carries `id`, `model`,
`display_name`, `created_at`, `updated_at`, `expires_at` (parsed `datetime` values),
and `usage.total_tokens`. Fields a provider omits remain `None`, and native fields
survive in `model_extra` and `model_dump()`. Gemini's deletion acknowledgement
returns the canonical cache ID without inventing a `deleted` flag.

## Pagination

Listing fetches exactly one page. Pass the opaque `next_cursor` to continue; `None`
marks the final page. Gemini maps `limit`/`cursor` to `page_size`/`page_token`, and
the SDK pager is never iterated, so later pages are not fetched automatically.

```python
page = provider.list_caches(limit=20)
if page.next_cursor is not None:
    next_page = provider.list_caches(limit=20, cursor=page.next_cursor)
```

## Provider options and errors

All cache methods accept `timeout`, `max_retries`, and `extra_headers`, with the
same meaning as for [Files](files.md#provider-options-and-errors). Other keyword
arguments, including native ones such as `system_instruction`, `page_size`, or
`kms_key_name`, raise `UnsupportedParameterError`.

Errors follow the provider's `unified_exceptions` option, falling back to the
`ANY_LLM_UNIFIED_EXCEPTIONS` environment variable. When enabled, a retrieve,
update, or delete HTTP 404 becomes `ProviderCacheNotFoundError`. The Gemini
Developer API answers an unknown or expired cache with HTTP 403 "CachedContent not
found (or permission denied)"; that response is `ProviderCacheNotFoundError` with
`status_code=403`, and a 403 without "not found" stays `AuthenticationError`.
