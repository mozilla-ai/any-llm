from typing import Any

from openai.types.model import Model as OpenAIModel
from pydantic import model_validator

# Where each listing states how many tokens a model's context holds, in the order
# they are read. vLLM and SGLang say ``max_model_len`` and report the length the
# engine was started with, which can be below what the model itself supports.
# OpenRouter and Together say ``context_length``.
_CONTEXT_WINDOW_KEYS = ("context_window", "max_model_len", "context_length")


def _positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


class Model(OpenAIModel):
    """A model as a provider lists it: the OpenAI fields, plus whatever else the provider sent."""

    context_window: int | None = None
    """How many tokens the model's context holds as this provider serves it, or ``None`` when the listing does not say."""

    @model_validator(mode="before")
    @classmethod
    def _read_context_window(cls, data: Any) -> Any:
        # The OpenAI SDK hands back its own Model, which this subclass would otherwise refuse.
        if isinstance(data, OpenAIModel):
            data = data.model_dump()
        # A malformed value leaves the field unset rather than failing the whole
        # listing, so one odd entry does not hide every other model.
        if not isinstance(data, dict):
            return data
        context_window = next(
            (found for key in _CONTEXT_WINDOW_KEYS if (found := _positive_int(data.get(key))) is not None), None
        )
        return {**data, "context_window": context_window}
