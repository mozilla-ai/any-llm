"""End-to-end tests for otari gateway built-in tools: web search and MCP execution.

These are otari-specific (not provider-parametrized): web search is the gateway-managed
``otari_web_search`` tool type, and MCP execution is driven by the top-level
``mcp_servers`` request field. Both run inside the gateway rather than the upstream
provider, so they only exist on otari.

The tests require a real otari endpoint. They run in platform mode against the hosted
gateway when ``OTARI_AI_TOKEN`` is set, or against a self-hosted deployment via
``OTARI_API_BASE`` (or the legacy ``GATEWAY_API_BASE``); otherwise they skip. CI provides
``OTARI_AI_TOKEN`` and lists otari in ``EXPECTED_PROVIDERS``, so they run there.
"""

import os

import pytest

from any_llm import AnyLLM, LLMProvider
from any_llm.types.completion import ChatCompletion

_OTARI_ENV_VARS = ("OTARI_AI_TOKEN", "OTARI_API_BASE", "GATEWAY_API_BASE")

# The hosted test account routes Anthropic completions reliably (mirrors tests/conftest.py).
_OTARI_MODEL = "anthropic:claude-haiku-4-5"

# Public streamable-HTTP MCP server used to exercise gateway MCP execution. Its tools are
# scoped to the Hugging Face Hub, so a model-lookup prompt forces a real tool call.
_MCP_SERVER = {"name": "huggingface", "url": "https://huggingface.co/mcp"}

pytestmark = pytest.mark.skipif(
    not any(os.getenv(var) for var in _OTARI_ENV_VARS),
    reason=f"otari endpoint not configured (set {' or '.join(_OTARI_ENV_VARS)})",
)


def _make_client() -> AnyLLM:
    # Let the provider auto-resolve auth: a platform token selects platform mode, otherwise
    # it falls back to OTARI_API_BASE/GATEWAY_API_BASE for a self-hosted deployment.
    return AnyLLM.create(LLMProvider.OTARI)


@pytest.mark.asyncio
async def test_otari_web_search_returns_answer() -> None:
    """tools=[{"type": "otari_web_search"}] runs the gateway web-search backend."""
    llm = _make_client()
    result = await llm.acompletion(
        model=_OTARI_MODEL,
        messages=[{"role": "user", "content": "Use web search to give me one news headline from today."}],
        tools=[{"type": "otari_web_search"}],
    )
    assert isinstance(result, ChatCompletion)
    content = result.choices[0].message.content
    assert content is not None
    assert content.strip()


@pytest.mark.asyncio
async def test_otari_mcp_execution_invokes_server_tool() -> None:
    """mcp_servers=[...] makes the gateway connect to the MCP server and run its tools."""
    llm = _make_client()
    result = await llm.acompletion(
        model=_OTARI_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Use the Hugging Face tools to look up the model 'bert-base-uncased' and report its full model id."
                ),
            }
        ],
        mcp_servers=[_MCP_SERVER],
        max_tool_iterations=5,
    )
    assert isinstance(result, ChatCompletion)
    content = result.choices[0].message.content
    assert content is not None
    # The HF MCP lookup tool resolves the canonical id (google-bert/bert-base-uncased);
    # a grounded answer echoes it.
    assert "bert-base-uncased" in content.lower()
