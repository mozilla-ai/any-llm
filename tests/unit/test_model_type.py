from typing import Any

import pytest
from openai.types.model import Model as OpenAIModel

from any_llm.types.model import Model


def _listing(**fields: Any) -> dict[str, Any]:
    return {"id": "m", "object": "model", "created": 0, "owned_by": "test", **fields}


def test_context_window_is_none_when_the_listing_does_not_say() -> None:
    assert Model(id="m", object="model", created=0, owned_by="test").context_window is None


def test_context_window_can_be_set_directly() -> None:
    assert Model(id="m", object="model", created=0, owned_by="test", context_window=8192).context_window == 8192


@pytest.mark.parametrize(
    ("key", "listing"),
    [
        ("max_model_len", "vLLM and SGLang"),
        ("context_length", "OpenRouter and Together"),
        ("context_window", "Groq"),
    ],
)
def test_context_window_is_read_from_each_listing_field(key: str, listing: str) -> None:
    model = Model.model_validate(_listing(**{key: 256000}))

    assert model.context_window == 256000, listing


def test_the_raw_field_is_kept_beside_the_normalized_one() -> None:
    model = Model.model_validate(_listing(max_model_len=256000))

    assert model.model_extra == {"max_model_len": 256000}


def test_context_window_prefers_an_explicit_value_to_an_engine_field() -> None:
    model = Model.model_validate(_listing(context_window=128000, max_model_len=256000, context_length=512000))

    assert model.context_window == 128000


def test_max_model_len_wins_over_context_length() -> None:
    model = Model.model_validate(_listing(max_model_len=256000, context_length=512000))

    assert model.context_window == 256000


@pytest.mark.parametrize("value", ["128k", 0, -1, True, 1.5, None, {"tokens": 1}])
def test_a_malformed_value_leaves_the_field_unset(value: object) -> None:
    model = Model.model_validate(_listing(max_model_len=value))

    assert model.context_window is None


def test_a_malformed_value_falls_through_to_the_next_field() -> None:
    model = Model.model_validate(_listing(context_window="128k", max_model_len=256000))

    assert model.context_window == 256000


def test_an_openai_sdk_model_validates_with_its_extra_fields() -> None:
    sdk_model = OpenAIModel.model_validate(_listing(max_model_len=256000))

    model = Model.model_validate(sdk_model)

    assert isinstance(model, Model)
    assert model.id == "m"
    assert model.context_window == 256000
    assert model.model_extra == {"max_model_len": 256000}


def test_a_model_validates_as_itself() -> None:
    model = Model.model_validate(_listing(max_model_len=256000))

    assert Model.model_validate(model) == model


def test_model_is_still_an_openai_model() -> None:
    assert isinstance(Model.model_validate(_listing()), OpenAIModel)
