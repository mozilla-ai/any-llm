import pytest
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.inception.inception import InceptionProvider
from any_llm.types.completion import CompletionParams


def test_inception_unsupported_response_format() -> None:
    class ResponseFormatModel(BaseModel):
        response: str

    params = CompletionParams(
        model_id="mercury", messages=[{"role": "user", "content": "Hello"}], response_format=ResponseFormatModel
    )
    with pytest.raises(UnsupportedParameterError, match="'response_format' is not supported for inception"):
        InceptionProvider._convert_completion_params(params)
