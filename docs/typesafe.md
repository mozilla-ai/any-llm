---
title: TypeSafe
description: Answer named questions about a state with TypeSafe System One
---

## Overview

[TypeSafe](https://docs.typesafe.ai/sdk/python) exposes System One, which answers named questions about a
state (text, a JSON object, or an array) instead of generating text. There are three question types:

- `Noul`: a yes/no question, answered with a probability between 0 and 1.
- `Choice`: pick one of the given criteria.
- `Score`: place the state on an ordered scale of criteria.

TypeSafe has no chat completions or embeddings. It implements the `decide` and `adecide` methods
(`SUPPORTS_DECIDE`), and `list_models` against the hosted API. Other providers serve the same API; see [Decide](decide.md).

## Installation

```bash
pip install any-llm-sdk[typesafe]
```

Set `TYPESAFE_API_KEY`, or pass `api_key` to `AnyLLM.create`.

## Quick Start

```python
from any_llm import AnyLLM
from any_llm.types.decide import Choice, Noul, Score

llm = AnyLLM.create("typesafe")

result = llm.decide(
    "jev-latest",
    "I was charged twice. Please fix this ASAP.",
    {
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
        "urgency": Score(instructions="How urgent is this?", criteria=["low", "medium", "high"]),
    },
)
print(result.nouls["billing"].noul, result.choices["tone"].choice, result.scores["urgency"].score)
```

`adecide` is the async version and takes the same arguments. The questions can also be dicts, such as
`{"type": "noul", "instructions": "Is this ticket about billing?"}`, or the question classes of `typesafe_sdk`.
The result is a `DecideResponse`, with the answers keyed by question name in `answers` and grouped by type in
`nouls`, `choices` and `scores`, and with the TypeSafe request ID in `id`.

The model you pass is always sent, so the SDK's `TYPESAFE_DEFAULT_MODEL` environment variable and the `model`
client argument do not apply to `decide`.

The top-level `decide` and `adecide` functions take a provider-prefixed model instead:

```python
from any_llm import decide
from any_llm.types.decide import Noul

result = decide("typesafe:jev-latest", "I was charged twice.", {"billing": Noul(instructions="Is this about billing?")})
print(result.nouls["billing"].noul)
```

All of them accept `timeout` (in seconds) and forward `retry`, `extra_headers` and `extra_body` to the
SDK's `AsyncTypeSafeClient.system_one`. `response_model` is not forwarded: `decide` always returns a
`DecideResponse`, and raises `UnsupportedParameterError` when it is passed.

## Self-hosted servers

Set `TYPESAFE_BASE_URL` (or pass `api_base`) to point the provider at a self-hosted server, such as a
local Ollama server that serves a TypeSafe model. Against any base URL other than the hosted API, the API
key is optional.

A self-hosted server usually does not serve the hosted model aliases, so pass a model it serves:

```python
from any_llm import AnyLLM
from any_llm.types.decide import Noul

llm = AnyLLM.create("typesafe", api_base="http://localhost:11434")
result = llm.decide("tev1:0.8b", "I was charged twice.", {"billing": Noul(instructions="Is this about billing?")})
print(result)
```

`list_models` only works against the hosted API, because the SDK expects its response shape.

## Errors

A question that is not valid, such as a `Score` with no criteria or a dict with an unknown `type`, raises
`pydantic.ValidationError` before anything is sent, with or without unified exceptions. The error names
every invalid question.

With `unified_exceptions=True`, TypeSafe errors are classified from the SDK's exception types before the
shared status and message rules run:

- A caller mistake the SDK catches before sending anything, such as an empty `questions` mapping or a
  state that cannot be encoded as JSON, raises `InvalidRequestError`. This includes a client setting the
  SDK rejects when the provider is created, such as an API key with whitespace inside it or a `timeout`
  that is not positive.
- A 400 or 422 raises `InvalidRequestError`, whatever the question names in its message say.
- A response the SDK cannot read with its own models raises `ProviderError`. Its `status_code` is the
  status the server sent, usually 200.
- A response the SDK reads but `DecideResponse` cannot, such as an answer with no `type`, raises
  `ProviderError` with or without unified exceptions.

Everything else, such as rate limits, authentication failures, and timeouts, follows the
[shared rules](api/exceptions.md).
