from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.deepseek.deepseek import DeepseekProvider
from any_llm.types.messages import MessageResponse
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [pytest.mark.asyncio, pytest.mark.parametrize("provider", [LLMProvider.DEEPSEEK])]


@pytest_asyncio.fixture
async def deepseek(
    provider: LLMProvider,
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> AsyncIterator[DeepseekProvider]:
    try:
        llm = DeepseekProvider(**provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("DeepSeek credentials missing: set DEEPSEEK_API_KEY")
    try:
        yield llm
    finally:
        await llm.client.close()
        await llm._anthropic_client.close()


async def test_deepseek_messages_accepts_claude_code_request_shape(deepseek: DeepseekProvider) -> None:
    """Claude Code sends betas and context_management on every request; neither may break it."""
    result = await deepseek.amessages(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=256,
        thinking={"type": "enabled", "budget_tokens": 1024},
        betas=["interleaved-thinking-2025-05-14"],
        context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
    )

    assert isinstance(result, MessageResponse)
    assert any(block.type == "thinking" for block in result.content)
    assert any(block.type == "text" and block.text for block in result.content)


async def test_deepseek_responses_runs_web_search_on_v4_pro(deepseek: DeepseekProvider) -> None:
    """DeepSeek runs a declared web_search tool itself on deepseek-v4-pro."""
    result = await deepseek.aresponses(
        "deepseek-v4-pro",
        input_data="Search the web for the current DeepSeek API pricing page and give me its title.",
        tools=[{"type": "web_search"}],
    )

    assert any(item.type == "web_search_call" for item in result.output)
