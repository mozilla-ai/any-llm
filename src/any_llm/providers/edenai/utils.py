"""Eden AI provider utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_llm.types.model import Model

if TYPE_CHECKING:
    from collections.abc import Sequence


def _convert_models_list(response: Any) -> Sequence[Model]:
    """Convert the Eden AI /v3/models response to valid Model objects.

    Eden AI's ``/v3/models`` response does not reliably populate every field the
    OpenAI ``Model`` schema requires (``created`` in particular is absent, and
    ``object``/``owned_by`` may be missing). The OpenAI SDK accepts the missing
    fields via ``model_construct()``, but the resulting objects fail round-trip
    serialization. This fills any missing required fields while preserving Eden
    AI's extra attributes (``model_name``, ``context_length``, ...).
    """
    raw_models = response.data if hasattr(response, "data") else response
    result: list[Model] = []
    for model in raw_models:
        data: dict[str, Any] = model.model_dump() if hasattr(model, "model_dump") else dict(vars(model))
        if data.get("object") is None:
            data["object"] = "model"
        if data.get("owned_by") is None:
            model_id = data.get("id") or ""
            data["owned_by"] = model_id.split("/", 1)[0] if "/" in model_id else "edenai"
        if data.get("created") is None:
            data["created"] = 0
        result.append(Model.model_validate(data))
    return result
