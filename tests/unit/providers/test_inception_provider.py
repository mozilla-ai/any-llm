import pytest
from pydantic import BaseModel

from any_llm import AnyLLM
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.inception.inception import InceptionProvider
from any_llm.types.completion import CompletionParams


def test_inception_warns_that_the_mercury_api_is_winding_down() -> None:
    with pytest.warns(FutureWarning, match="after December 28, 2026") as record:
        InceptionProvider(api_key="test-key")
    assert record[0].filename == __file__


def test_inception_wind_down_warning_reaches_anyllm_create() -> None:
    with pytest.warns(FutureWarning, match="docs.inceptionlabs.ai/support/wind-down"):
        AnyLLM.create("inception", api_key="test-key")


def test_inception_unsupported_response_format() -> None:
    class ResponseFormatModel(BaseModel):
        response: str

    params = CompletionParams(
        model_id="mercury", messages=[{"role": "user", "content": "Hello"}], response_format=ResponseFormatModel
    )
    with pytest.raises(UnsupportedParameterError, match="'response_format' is not supported for inception"):
        InceptionProvider._convert_completion_params(params)


def test_inception_remaps_max_tokens_to_max_completion_tokens() -> None:
    params = CompletionParams(model_id="mercury", messages=[{"role": "user", "content": "Hello"}], max_tokens=8192)
    result = InceptionProvider._convert_completion_params(params)
    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 8192
