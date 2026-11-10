"""OpenRouter Provider Utilities.

Utilities for building OpenRouter-specific request parameters, with explicit
opt-in handling for reasoning capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_llm.types.model import Model
from any_llm.types.rerank import RerankMeta, RerankResponse, RerankResult, RerankUsage

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
    resulting objects fail round-trip serialization. This function fills the
    missing required fields while preserving any extra OpenRouter attributes
    (e.g. ``name``, ``pricing``) that may already be present on the SDK objects.
    """
    raw_models = response.data if hasattr(response, "data") else response
    result: list[Model] = []
    for model in raw_models:
        data: dict[str, Any] = model.model_dump() if hasattr(model, "model_dump") else dict(vars(model))
        if data.get("object") is None:
            data["object"] = "model"
        if data.get("owned_by") is None:
            data["owned_by"] = "openrouter"
        if data.get("created") is None:
            data["created"] = 0
        result.append(Model.model_validate(data))
    return result


def convert_rerank_response(response: dict[str, Any]) -> RerankResponse:
    """Convert an OpenRouter `/rerank` JSON body to a normalized RerankResponse.

    The echoed `document` on each result is dropped, as RerankResult carries none. `search_units`
    goes to `meta.billed_units`, where Cohere reports it.
    """
    results = [
        RerankResult(index=item["index"], relevance_score=item["relevance_score"]) for item in response["results"]
    ]
    results.sort(key=lambda r: r.relevance_score, reverse=True)

    usage = response.get("usage") or {}
    search_units = usage.get("search_units")
    return RerankResponse(
        id=response.get("id"),
        model=response.get("model"),
        results=results,
        meta=RerankMeta(billed_units={"search_units": float(search_units)}) if search_units is not None else None,
        usage=RerankUsage(total_tokens=usage["total_tokens"]) if usage.get("total_tokens") is not None else None,
    )


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
