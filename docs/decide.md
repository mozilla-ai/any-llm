---
title: Decide
description: Answer named questions about a state with any provider that serves System One
---

## Overview

The decide API answers named questions about a state (text, a JSON object, or an array) instead of
generating text. TypeSafe defined the request shape (System One), and other providers serve the same
shape, so one call works with each of them. There are three question types:

- `Noul`: a yes/no question, answered with a probability between 0 and 1.
- `Choice`: pick one of the given criteria.
- `Score`: place the state on an ordered scale of criteria.

## Supported Providers

| Provider | `SUPPORTS_DECIDE` | Notes |
|----------|-------------------|-------|
| OpenRouter | Yes | `POST /api/alpha/decisions` through the OpenAI client. Use an OpenRouter model ID, such as `typesafe/jev-1.13`. |
| TypeSafe | Yes | `POST /v1/systemone` through the TypeSafe SDK. See [TypeSafe](typesafe.md). |

All other providers return `SUPPORTS_DECIDE = False` and raise `NotImplementedError` if called.

## Quick Start

```python
from any_llm import decide
from any_llm.types.decide import Choice, Noul

result = decide(
    "openrouter:typesafe/jev-1.13",
    "I was charged twice. Please fix this ASAP.",
    {
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
    },
)
print(result.nouls["billing"].noul, result.choices["tone"].choice)
```

`adecide` is the async version and takes the same arguments. Change the prefix to change the provider,
for example `typesafe:jev-latest`.

## Request options

For OpenRouter, `timeout` (in seconds) and `extra_headers` apply to the HTTP request. Other keyword
arguments, and the keys of `extra_body`, are added to the request body, so OpenRouter fields such as
`session_id` or `provider` go through unchanged.
