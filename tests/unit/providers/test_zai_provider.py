import pytest
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.zai.zai import ZaiProvider
from any_llm.types.completion import CompletionParams


def test_zai_unsupported_response_format() -> None:
    class ResponseFormatModel(BaseModel):
        response: str

    params = CompletionParams(
        model_id="zai-model", messages=[{"role": "user", "content": "Hello"}], response_format=ResponseFormatModel
    )
    with pytest.raises(UnsupportedParameterError, match="'response_format' is not supported for zai"):
        ZaiProvider._convert_completion_params(params)


def test_zai_remaps_max_tokens_to_max_completion_tokens() -> None:
    params = CompletionParams(model_id="zai-model", messages=[{"role": "user", "content": "Hello"}], max_tokens=8192)
    result = ZaiProvider._convert_completion_params(params)
    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 8192
