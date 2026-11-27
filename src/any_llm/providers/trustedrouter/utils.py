"""Utilities for TrustedRouter model listing and privacy-floor routing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from any_llm.types.model import Model

VALID_MIN_PRIVACY = frozenset({"any", "no_store", "zdr", "confidential", "e2e", "e2ee"})
_HIDDEN_MODEL_FLAGS = ("internal_only", "configuration_hidden", "synthetic_monitor")


def normalize_min_privacy(min_privacy: str) -> str:
    """Validate and normalize a TrustedRouter ``min_privacy`` privacy tier."""
    if not isinstance(min_privacy, str):
        msg = f"min_privacy must be one of {sorted(VALID_MIN_PRIVACY)}, got {type(min_privacy).__name__}"
        raise ValueError(msg)
    normalized = min_privacy.strip().lower()
    if normalized not in VALID_MIN_PRIVACY:
        msg = f"unsupported min_privacy {min_privacy!r}; expected one of {sorted(VALID_MIN_PRIVACY)}"
        raise ValueError(msg)
    return normalized


def merge_min_privacy(
    kwargs: dict[str, Any],
    *,
    min_privacy: str | None,
    default_min_privacy: str | None,
) -> None:
    """Merge ``min_privacy`` into ``kwargs['extra_body']['provider']`` without mutating caller dicts.

    An explicit per-request ``min_privacy`` overrides any value in ``extra_body['provider']``;
    an instance-level default only fills in ``min_privacy`` when the request did not already
    specify one in ``extra_body['provider']``.
    """
    explicit = normalize_min_privacy(min_privacy) if min_privacy is not None else None
    effective = explicit or default_min_privacy
    if effective is None:
        return

    existing_extra_body = kwargs.get("extra_body")
    if existing_extra_body is not None and not isinstance(existing_extra_body, Mapping):
        msg = "extra_body must be a mapping when combining with min_privacy"
        raise ValueError(msg)
    extra_body = dict(existing_extra_body) if existing_extra_body is not None else {}

    existing_provider = extra_body.get("provider")
    if existing_provider is not None and not isinstance(existing_provider, Mapping):
        msg = "extra_body['provider'] must be a mapping when combining with min_privacy"
        raise ValueError(msg)
    provider_prefs = dict(existing_provider) if existing_provider is not None else {}

    if explicit is not None:
        provider_prefs["min_privacy"] = explicit
    else:
        provider_prefs.setdefault("min_privacy", effective)

    extra_body["provider"] = provider_prefs
    kwargs["extra_body"] = extra_body


def _model_to_dict(model: Any) -> dict[str, Any]:
    if isinstance(model, BaseModel):
        return model.model_dump()
    if isinstance(model, Mapping):
        return dict(model)
    return dict(vars(model))


def _is_hidden_model(data: Mapping[str, Any]) -> bool:
    tr_meta = data.get("trustedrouter")
    if not isinstance(tr_meta, Mapping):
        return False
    return any(bool(tr_meta.get(flag)) for flag in _HIDDEN_MODEL_FLAGS)


def _convert_models_list(response: Any) -> Sequence[Model]:
    """Convert TrustedRouter ``/v1/models`` entries to valid ``Model`` objects.

    Filters out internal, hidden, and synthetic-monitor catalog entries flagged in
    ``trustedrouter`` metadata, and populates ``object``, ``owned_by`` (from
    ``trustedrouter.provider`` or ``"trustedrouter"``), and ``created`` when omitted
    while preserving extra fields such as pricing, architecture, and routing aliases.
    """
    raw_models = response.data if hasattr(response, "data") else response
    result: list[Model] = []
    for item in raw_models:
        data = _model_to_dict(item)
        if _is_hidden_model(data):
            continue
        tr_meta = data.get("trustedrouter")
        upstream_provider = tr_meta.get("provider") if isinstance(tr_meta, Mapping) else None
        if data.get("object") is None:
            data["object"] = "model"
        if data.get("owned_by") is None:
            data["owned_by"] = (
                upstream_provider if isinstance(upstream_provider, str) and upstream_provider else "trustedrouter"
            )
        if data.get("created") is None:
            data["created"] = 0
        result.append(Model.model_validate(data))
    return result
