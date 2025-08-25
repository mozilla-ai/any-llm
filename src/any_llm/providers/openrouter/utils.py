"""OpenRouter Provider Utilities.

Utilities for building OpenRouter-specific request parameters, with explicit
opt-in handling for reasoning capabilities.
"""

from __future__ import annotations

from typing import Any


def build_reasoning_directive(
    *,
    reasoning: Any | None = None,
    include_reasoning: bool | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any] | None:
    """Build OpenRouter's reasoning directive from user parameters.

    Args:
        reasoning: Direct reasoning config (dict/object) - preferred
        include_reasoning: Boolean toggle for reasoning output
        reasoning_effort: Effort level ("low"/"medium"/"high")

    Returns:
        Dict for request body "reasoning" key, or None to omit
    """
    if reasoning is not None:
        return _normalize_reasoning_obj(reasoning)

    if include_reasoning is True:
        return {}
    if include_reasoning is False:
        return {"exclude": True}

    if reasoning_effort:
        level = str(reasoning_effort).lower()
        if level in {"low", "medium", "high"}:
            return {"effort": level}

    return None


def _normalize_reasoning_obj(obj: Any) -> dict[str, Any]:
    """Normalize reasoning config to OpenRouter format."""

    def _get(o: Any, k: str) -> Any:
        return o.get(k) if isinstance(o, dict) else getattr(o, k, None)

    out: dict[str, Any] = {}

    effort = _get(obj, "effort")
    if effort is not None:
        out["effort"] = str(effort).lower()

    max_tokens = _get(obj, "max_tokens")
    if max_tokens is not None:
        out["max_tokens"] = int(max_tokens)

    exclude = _get(obj, "exclude")
    if exclude is not None:
        out["exclude"] = bool(exclude)

    enabled = _get(obj, "enabled")
    if enabled is not None:
        out["enabled"] = bool(enabled)

    return out
