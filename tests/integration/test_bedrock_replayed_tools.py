from typing import Any

import pytest

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [pytest.mark.asyncio, pytest.mark.parametrize("provider", [LLMProvider.BEDROCK])]


async def test_bedrock_answers_a_replayed_tool_turn_without_tools(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """An agent loop that withholds its tools to make the model answer still sends the calls it made."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Bedrock credentials missing")

    result = await llm.acompletion(
        model=provider_model_map[provider],
        max_tokens=100,
        messages=[
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature": "15C", "condition": "sunny"}'},
            {"role": "user", "content": "No more tool calls are available. Answer now from the result you have."},
        ],
    )

    assert result.choices[0].message.content
