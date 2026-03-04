from dataclasses import dataclass

from any_llm.providers.portkey.portkey import PortkeyProvider
from any_llm.types.completion import CompletionParams


def test_convert_completion_params_with_dataclass_response_format() -> None:
    """Test that dataclass response_format is converted to JSON schema format."""

    @dataclass
    class TestOutput:
        name: str
        value: int

    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        response_format=TestOutput,
    )

    result = PortkeyProvider._convert_completion_params(params)

    assert "response_format" in result
    assert result["response_format"]["type"] == "json_schema"
    assert result["response_format"]["json_schema"]["name"] == "response_schema"
    assert "properties" in result["response_format"]["json_schema"]["schema"]
    assert "name" in result["response_format"]["json_schema"]["schema"]["properties"]
    assert "value" in result["response_format"]["json_schema"]["schema"]["properties"]
