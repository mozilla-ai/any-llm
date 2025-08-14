from typing import Any

import pytest
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig


def _mk_provider() -> Any:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    return CohereProvider(ApiConfig(api_key="test-api-key"))


def test_preprocess_response_format() -> None:
    provider = _mk_provider()

    class StructuredOutput(BaseModel):
        foo: str
        bar: int

    json_schema = {"type": "json_object", "schema": StructuredOutput.model_json_schema()}

    # input BaseModel should output a dict
    outp_basemodel = provider._preprocess_response_format(StructuredOutput)

    # input dict should output a dict
    outp_dict = provider._preprocess_response_format(json_schema)

    # both should output a dict
    assert isinstance(outp_basemodel, dict)
    assert isinstance(outp_dict, dict)

    # should equal each other
    assert outp_basemodel == outp_dict


def test_stream_and_response_format_combination_raises() -> None:
    provider = _mk_provider()

    with pytest.raises(UnsupportedParameterError):
        provider.completion(
            model="model-id",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            response_format={"type": "json_object"},
        )


def test_parallel_tool_calls_raises() -> None:
    provider = _mk_provider()

    with pytest.raises(UnsupportedParameterError):
        provider.completion(
            model="model-id",
            messages=[{"role": "user", "content": "Hello"}],
            parallel_tool_calls=False,
        )
