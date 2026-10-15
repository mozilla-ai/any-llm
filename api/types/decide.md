---
title: Decide Types
description: Question and answer models for the decide API
---

The questions passed to `any_llm.decide()` and `any_llm.adecide()`, and the response they return.

**Import:** `from any_llm.types.decide import Noul, Choice, Score, DecideResponse`

## Questions

Each question is identified by its `type`, so a question can also be passed as a dict with a `type` key.

### `Noul`

A yes/no question, answered with the probability of yes.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal['noul']` |  |
| `instructions` | `str \| Mapping[str, Any] \| Sequence[Any] \| None` | The question to ask; optional. |
| `criteria` | `NoulCriteria \| None` | Optional descriptions of the yes and no outcomes. |

### `NoulCriteria`

Optional descriptions of the yes and no outcomes of a `Noul` question.

| Field | Type | Description |
|-------|------|-------------|
| `true` | `str \| Mapping[str, Any] \| Sequence[Any] \| None` | Description of the yes outcome; `None` leaves it undescribed. |
| `false` | `str \| Mapping[str, Any] \| Sequence[Any] \| None` | Description of the no outcome; `None` leaves it undescribed. |

### `Choice`

A question that picks one of the named criteria.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal['choice']` |  |
| `instructions` | `str \| Mapping[str, Any] \| Sequence[Any] \| None` | The question to ask; optional. |
| `criteria` | `Mapping[str, str \| Mapping[str, Any] \| Sequence[Any] \| None]` | Choice names mapped to a description of when each applies, or None to use the name alone |

### `Score`

A question that rates the state on an ordered scale.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal['score']` |  |
| `instructions` | `str \| Mapping[str, Any] \| Sequence[Any] \| None` | The question to ask; optional. |
| `criteria` | `Sequence[str \| Mapping[str, Any] \| Sequence[Any]]` | Ordered score levels; each level's position is its score, starting at zero |

## Answers

### `NoulAnswer`

The answer to a `Noul` question.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal['noul']` |  |
| `noul` | `float` | Probability of yes, from 0 to 1 |

### `ChoiceAnswer`

The answer to a `Choice` question.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal['choice']` |  |
| `choice` | `str` | The most likely choice name |
| `confidence` | `float` | Confidence in the choice, from 0 to 1 |
| `probabilities` | `dict[str, float]` | Probability of each choice name |

### `ScoreAnswer`

The answer to a `Score` question.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal['score']` |  |
| `score` | `float` | Probability-weighted average of the score levels |
| `confidence` | `float` | Confidence in the score, from 0 to 1 |
| `legend` | `dict[int, str \| Mapping[str, Any] \| Sequence[Any]]` | Score levels mapped to the criteria that define them |
| `probabilities` | `dict[int, float]` | Probability of each score level |

## Response

### `DecideResponse`

Answers keyed by the question names of the request, provider-agnostic.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str \| None` | Provider-assigned request ID, when the provider reports one |
| `model` | `str \| None` | Model that served the request: the provider-reported one when available, else the requested model |
| `answers` | `dict[str, NoulAnswer \| ChoiceAnswer \| ScoreAnswer]` | One answer per question, keyed by question name |
| `usage` | `DecideUsage \| None` | Token usage, when the provider reports it. |

### `DecideUsage`

Token usage of a decide request.

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | `int \| None` | Input tokens used, or `None` when the provider did not report it. |
| `output_tokens` | `int \| None` | Output tokens used, or `None` when the provider did not report it. |

`DecideResponse` also exposes the answers grouped by question type as `nouls`, `choices` and `scores`.
