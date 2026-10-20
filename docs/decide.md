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
| llama.cpp | Yes | `POST /v1/systemone` on llama-server (build b11371 or newer) through the OpenAI client. The server must load a decision model; with one model loaded, the `model` you pass is ignored. Set `api_base` (or `LLAMACPP_API_BASE`) to the server's `/v1` URL; the default is `http://127.0.0.1:8080/v1`. No API key is needed. |
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

## Response

The result is a `DecideResponse`. `answers` holds the answers keyed by question name, and `nouls`,
`choices` and `scores` group them by type. `usage` holds the token counts and, when the provider
reports one, `cost`: what the provider charged in USD. OpenRouter reports it.

The response keeps every field the provider sends, including fields any-llm does not model, such as
OpenRouter's `provider`. Read them from `model_extra`, or get the whole response with `model_dump()`.
An answer of a type any-llm does not model, such as OpenRouter's `box`, is kept as an `UnknownAnswer`
with all its fields. It appears in `answers`, but not in `nouls`, `choices` or `scores`.

```python
from any_llm import decide
from any_llm.types.decide import Noul

result = decide("openrouter:typesafe/jev-1.13", "I was charged twice.", {"billing": Noul(instructions="Billing?")})
print(result.model_dump(exclude_none=True))
```

The TypeSafe SDK logs a warning when it ignores an answer type it does not know. any-llm reads the
response body itself, so it still keeps that answer.

## Request options

For OpenRouter and llama.cpp, `timeout` (in seconds) and `extra_headers` apply to the HTTP request.
Other keyword arguments, and the keys of `extra_body`, are added to the request body, so OpenRouter
fields such as `session_id` or `provider` go through unchanged.

## Images

Pass `images`, a list of data URLs (`data:image/...;base64,...`), when the questions are about a picture.
Only providers with `SUPPORTS_DECIDE_IMAGE` accept them: llama.cpp, with a vision decision model.
OpenRouter and TypeSafe raise `UnsupportedParameterError` before anything is sent, because their APIs take
text only.

```python
import base64

from any_llm import decide
from any_llm.types.decide import Noul

image = "data:image/png;base64," + base64.b64encode(b"<png bytes>").decode()
result = decide(
    "llamacpp:laya",
    "A screenshot of a support ticket.",
    {"billing": Noul(instructions="Is this ticket about billing?")},
    images=[image],
)
print(result.nouls["billing"].noul)
```

## llama.cpp

To reach a llama-server on another host, pass its `/v1` URL:

```python
from any_llm import decide
from any_llm.types.decide import Noul

result = decide(
    "llamacpp:laya",
    "I was charged twice.",
    {"billing": Noul(instructions="Is this ticket about billing?")},
    api_base="http://gpu-box:8080/v1",
)
print(result.nouls["billing"].noul)
```
