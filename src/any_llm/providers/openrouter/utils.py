"""OpenRouter Provider Utilities.

Utilities for building OpenRouter-specific request parameters, with explicit
opt-in handling for reasoning capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_llm.types.model import Model

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_reasoning_directive(
    *,
    reasoning: Any | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any] | None:
    """Build OpenRouter's reasoning directive from user parameters.

    Args:
        reasoning: Direct reasoning config (dict/object) for advanced control
        reasoning_effort: Standard effort level ("low"/"medium"/"high")

    Returns:
        Dict for request body "reasoning" key, or None to omit
    """
    # Priority 1: Direct reasoning object for advanced users
    if reasoning is not None:
        return _normalize_reasoning_obj(reasoning)

    # Priority 2: Standard reasoning_effort parameter
    if reasoning_effort and reasoning_effort not in ("auto", "none"):
        level = str(reasoning_effort).lower()
        if level in {"low", "medium", "high"}:
            return {"effort": level}

    # Default: No reasoning (including for "auto")
    return None


def _convert_models_list(response: Any) -> Sequence[Model]:
    """Convert OpenRouter list models response to valid Model objects.

    OpenRouter's ``/api/v1/models`` endpoint omits the ``object`` and
    ``owned_by`` fields that the OpenAI ``Model`` schema requires. The OpenAI
    SDK silently accepts those missing fields via ``model_construct()``, but the
    resulting objects fail round-trip serialization. This function rebuilds each
    model with proper defaults so downstream consumers always receive
    spec-compliant ``Model`` instances.
    """
    raw_models = response.data if hasattr(response, "data") else response
    return [
        Model(
            id=model.id,
            created=model.created if model.created is not None else 0,
            object="model",
            owned_by=model.owned_by if model.owned_by is not None else "openrouter",
        )
        for model in raw_models
    ]


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
