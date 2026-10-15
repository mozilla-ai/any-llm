---
title: Decide
description: Answer named yes/no, choice and score questions about a state
---

The `decide` and `adecide` functions answer named questions about a state (text, a JSON object, or an array) with a provider that supports the decide API, such as [TypeSafe](../typesafe.md).

## `any_llm.decide()`

```
def decide(
    model: str,
    state: str | Mapping[str, Any] | Sequence[Any],
    questions: Mapping[str, Noul | Choice | Score | Mapping[str, Any]],
    *,
    provider: str | LLMProvider | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DecideResponse
```

## `any_llm.adecide()`

Async variant with the same parameters.

```
async def adecide(
    model: str,
    state: str | Mapping[str, Any] | Sequence[Any],
    questions: Mapping[str, Noul | Choice | Score | Mapping[str, Any]],
    *,
    provider: str | LLMProvider | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DecideResponse
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | Provider-prefixed model ID (e.g., "typesafe:jev-latest"). |
| `state` | `str \| Mapping[str, Any] \| Sequence[Any]` | *required* | Text, a JSON object, or an array the questions refer to. |
| `questions` | `Mapping[str, Noul \| Choice \| Score \| Mapping[str, Any]]` | *required* | Names mapped to `Noul`, `Choice` or `Score` questions, their dict forms, or objects with the same attributes. |
| `provider` | `str \| LLMProvider \| None` | None | Provider name or enum. Inferred from model string if omitted. |
| `api_key` | `str \| None` | None | Provider API key. Falls back to environment variable. |
| `api_base` | `str \| None` | None | Provider API base URL. Falls back to environment variable. |
| `client_args` | `dict[str, Any] \| None` | None | Additional arguments passed to the provider client constructor. |
| `**kwargs` | `Any` | *required* | Additional arguments for the call, such as `timeout` in seconds. |

## Return Value

Returns a [`DecideResponse`](types/decide.md) containing:

- `answers` -- one answer per question, keyed by question name.
- `nouls`, `choices`, `scores` -- the answers grouped by question type.
- `model` -- the model that served the request.
- `usage` -- token usage, when the provider reports it.
- `id` -- the provider's request ID, when it reports one.

## Usage

### Every question type

```python
from any_llm import decide
from any_llm.types.decide import Choice, Noul, Score

result = decide(
    "typesafe:jev-latest",
    "I was charged twice. Please fix this ASAP.",
    {
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
        "urgency": Score(instructions="How urgent is this?", criteria=["low", "medium", "high"]),
    },
)

print(result.nouls["billing"].noul)
print(result.choices["tone"].choice)
print(result.scores["urgency"].score)
```

### Async

```python
import asyncio
from any_llm import adecide
from any_llm.types.decide import Noul

async def main():
    result = await adecide(
        "typesafe:jev-latest",
        "I was charged twice.",
        {"billing": Noul(instructions="Is this ticket about billing?")},
    )
    print(result.nouls["billing"].noul)

asyncio.run(main())
```

{% hint style="info" %}
Not all providers support decide. Check the [providers page](../providers.md) for support details, or query `ProviderMetadata.decide` programmatically. A provider that does not support it raises `NotImplementedError`.
{% endhint %}
