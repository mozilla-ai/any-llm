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

TypeSafe has no chat completions or embeddings, so the provider adds its own `decide` and `adecide`
methods. It also supports `list_models` against the hosted API.

## Installation

```bash
pip install any-llm-sdk[typesafe]
```

Set `TYPESAFE_API_KEY`, or pass `api_key` to `AnyLLM.create`.

## Quick Start

```python
from typesafe_sdk import Choice, Noul, Score

from any_llm import AnyLLM

llm = AnyLLM.create("typesafe")

result = llm.decide(
    "I was charged twice. Please fix this ASAP.",
    {
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
        "urgency": Score(instructions="How urgent is this?", criteria=["low", "medium", "high"]),
    },
)
print(result)
```

`adecide` is the async version and takes the same arguments. Both accept `model`, `response_model`
(a Pydantic model to parse the answers into), and the other keyword arguments of the SDK's
`AsyncTypeSafeClient.system_one`, such as `timeout`.

## Self-hosted servers

Set `TYPESAFE_BASE_URL` (or pass `api_base`) to point the provider at a self-hosted server, such as a
local Ollama server that serves a TypeSafe model. Against any base URL other than the hosted API, the API
key is optional.

A self-hosted server usually does not serve the SDK's default model alias, so pass `model` on each
call or set `TYPESAFE_DEFAULT_MODEL`:

```python
from typesafe_sdk import Noul

from any_llm import AnyLLM

llm = AnyLLM.create("typesafe", api_base="http://localhost:11434")
result = llm.decide(
    "I was charged twice.", {"billing": Noul(instructions="Is this about billing?")}, model="tev1:0.8b"
)
print(result)
```

`list_models` only works against the hosted API, because the SDK expects its response shape.
