---
title: Prompt Caching
description: Cache a Gemini or Vertex AI prompt prefix by marking it with Anthropic-style cache_control
---

# Prompt Caching

Gemini and Vertex AI turn Anthropic-style `cache_control` breakpoints into
[context caches](https://ai.google.dev/gemini-api/docs/caching). Mark the end of
the prefix you want cached and any-llm creates the cache, reuses it on later
calls, and reports the tokens it wrote and read. There is no cache API to call.

```python
from any_llm import AnyLLM

provider = AnyLLM.create("gemini")
manual = open("manual.txt").read()  # needs at least 1,024 tokens, see below

response = provider.completion(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "system",
            "content": [{"type": "text", "text": manual, "cache_control": {"type": "ephemeral"}}],
        },
        {"role": "user", "content": "What does chapter 3 say about retries?"},
    ],
)
print(response.usage.prompt_tokens_details)
```

## Where a breakpoint can go

- **Chat Completions**: a `cache_control` dict on a message or on any of its
  content parts, the convention LiteLLM and OpenRouter use for OpenAI-format
  requests.
- **Messages** (`provider.messages(...)`): `cache_control` on system blocks, on
  user, assistant, and `tool_result` blocks, and the top-level `cache_control`
  parameter, which marks the last message as Anthropic's automatic caching does.

The value is `{"type": "ephemeral"}`, optionally with `"ttl": "5m"` or `"ttl": "1h"`.
The Messages bridge does not forward these markers to other providers.

## What gets cached

Gemini caches a single prefix, so **the last breakpoint wins**. A request that
uses a cache cannot also send a system instruction or tools, so the cache holds
the system instruction, the tools and tool choice, and the conversation up to and
including the marked message. The request then sends only the messages after it.
Gemini rejects a request with no contents, so when the breakpoint is on the last
message that message stays outside the cache.

The cache lives for 5 minutes, or 1 hour if any breakpoint in the prefix asks for
`"1h"`. Caches are reused within the process across provider instances, keyed by
the credentials, endpoint, model, and cached content, and are recreated shortly
before they expire.

Gemini refuses caches below a minimum size of 1,024 to 4,096 tokens, depending on
the model. A refused prefix is sent uncached, and any-llm does not try to create
it again until the TTL passes. Any other failure to create a cache also falls back
to an uncached request. Passing your own `cached_content` turns the automatic path
off for that call.

## Usage reporting

On the call that creates the cache, the cached prefix is reported as
`usage.prompt_tokens_details.cache_write_tokens` (with `cached_tokens=0`). Later
calls that reuse it report `cached_tokens`. Both are subsets of `prompt_tokens`.
Through `messages()` they appear as `cache_creation_input_tokens` and
`cache_read_input_tokens`.

## Implicit caching

Gemini 2.5 and later models also cache repeated prefixes implicitly, with no
markers and no storage cost. Those hits are reported as `cached_tokens` too.
Breakpoints are worth adding when you want a guaranteed discount on a large,
stable prefix.
