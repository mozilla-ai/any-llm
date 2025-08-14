import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig
from any_llm.providers.llamafile.llamafile import LlamafileProvider
from any_llm.types.completion import CompletionParams


def test_response_format_dict_raises() -> None:
    provider = LlamafileProvider(ApiConfig())
    with pytest.raises(UnsupportedParameterError):
        provider.completion(
            CompletionParams(
                model_id="llama3.1",
                messages=[{"role": "user", "content": "Hi"}],
                response_format={"type": "json_object"},
            )
        )


def test_tools_raises() -> None:
    provider = LlamafileProvider(ApiConfig())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "foo",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    with pytest.raises(UnsupportedParameterError):
        provider.completion(
            CompletionParams(
                model_id="llama3.1",
                messages=[{"role": "user", "content": "Hi"}],
                tools=tools,
            )
        )
