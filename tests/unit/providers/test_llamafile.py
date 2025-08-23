from unittest.mock import patch

import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig
from any_llm.providers.llamafile.llamafile import LlamafileProvider
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import CompletionParams


@pytest.mark.asyncio
async def test_response_format_dict_raises() -> None:
    provider = LlamafileProvider(ApiConfig())
    with pytest.raises(UnsupportedParameterError):
        await provider.acompletion(
            CompletionParams(
                model_id="llama3.1",
                messages=[{"role": "user", "content": "Hi"}],
                response_format={"type": "json_object"},
            )
        )


@pytest.mark.asyncio
async def test_calls_completion() -> None:
    provider = LlamafileProvider(ApiConfig())
    params = CompletionParams(model_id="llama3.1", messages=[{"role": "user", "content": "Hi"}])
    sentinel = object()
    with patch.object(BaseOpenAIProvider, "acompletion", autospec=True, return_value=sentinel) as mock_super:
        result = await provider.acompletion(params, temperature=0.1)
        assert result is sentinel
        mock_super.assert_called_once_with(provider, params, temperature=0.1)


@pytest.mark.asyncio
async def test_tools_raises() -> None:
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
        await provider.acompletion(
            CompletionParams(
                model_id="llama3.1",
                messages=[{"role": "user", "content": "Hi"}],
                tools=tools,
            )
        )
