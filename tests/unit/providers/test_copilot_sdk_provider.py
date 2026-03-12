"""Unit tests for CopilotSdkProvider."""
from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.providers.copilot_sdk.copilot_sdk import (
    CopilotSdkProvider,
    _build_chat_completion,
    _copilot_model_to_openai,
    _messages_to_prompt,
)
from any_llm.types.completion import CompletionParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(**kwargs: Any) -> CopilotSdkProvider:
    """Instantiate CopilotSdkProvider while bypassing real CLI startup."""
    p = object.__new__(CopilotSdkProvider)
    p._resolved_token = kwargs.get("api_key")
    p._cli_url = kwargs.get("api_base")
    p._cli_path = None
    p._extra_kwargs = {}
    p._copilot_client = None
    p._client_lock = asyncio.Lock()
    return p


def _make_model_info(model_id: str = "gpt-4o", name: str = "GPT-4o") -> Any:
    """Return a minimal mock ModelInfo."""
    m = MagicMock()
    m.id = model_id
    m.name = name
    return m


# ---------------------------------------------------------------------------
# _messages_to_prompt
# ---------------------------------------------------------------------------

def test_messages_to_prompt_simple_user() -> None:
    msgs = [{"role": "user", "content": "Hello"}]
    assert _messages_to_prompt(msgs) == "User: Hello"


def test_messages_to_prompt_system_prepended() -> None:
    msgs = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
    ]
    result = _messages_to_prompt(msgs)
    assert result.startswith("Be concise.")
    assert "User: Hi" in result


def test_messages_to_prompt_multi_turn() -> None:
    msgs = [
        {"role": "user", "content": "Ping"},
        {"role": "assistant", "content": "Pong"},
        {"role": "user", "content": "Again"},
    ]
    result = _messages_to_prompt(msgs)
    assert "User: Ping" in result
    assert "Assistant: Pong" in result
    assert "User: Again" in result


def test_messages_to_prompt_multimodal_extracts_text() -> None:
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Tell me"}, {"type": "image_url"}]}]
    result = _messages_to_prompt(msgs)
    assert "Tell me" in result


# ---------------------------------------------------------------------------
# _build_chat_completion
# ---------------------------------------------------------------------------

def test_build_chat_completion_structure() -> None:
    cc = _build_chat_completion("Hello world", "gpt-4o")
    assert cc.choices[0].message.content == "Hello world"
    assert cc.model == "gpt-4o"
    assert cc.choices[0].finish_reason == "stop"
    assert cc.object == "chat.completion"


# ---------------------------------------------------------------------------
# _copilot_model_to_openai
# ---------------------------------------------------------------------------

def test_copilot_model_to_openai_maps_id() -> None:
    info = _make_model_info("claude-sonnet-4-5", "Claude Sonnet 4.5")
    model = _copilot_model_to_openai(info)
    assert model.id == "claude-sonnet-4-5"
    assert model.owned_by == "github-copilot"
    assert model.object == "model"


# ---------------------------------------------------------------------------
# Provider class attributes
# ---------------------------------------------------------------------------

def test_provider_required_attributes() -> None:
    assert CopilotSdkProvider.PROVIDER_NAME == "copilot_sdk"
    assert CopilotSdkProvider.SUPPORTS_COMPLETION is True
    assert CopilotSdkProvider.SUPPORTS_LIST_MODELS is True
    assert CopilotSdkProvider.SUPPORTS_EMBEDDING is False
    assert CopilotSdkProvider.SUPPORTS_COMPLETION_STREAMING is False
    assert CopilotSdkProvider.MISSING_PACKAGES_ERROR is None


# ---------------------------------------------------------------------------
# Auth: _verify_and_set_api_key
# ---------------------------------------------------------------------------

def test_api_key_explicit_wins() -> None:
    p = object.__new__(CopilotSdkProvider)
    with patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": "env-token"}, clear=False):
        result = p._verify_and_set_api_key("explicit-token")
    assert result == "explicit-token"


def test_api_key_falls_back_to_copilot_env() -> None:
    p = object.__new__(CopilotSdkProvider)
    with patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": "copilot-env"}, clear=False):
        result = p._verify_and_set_api_key(None)
    assert result == "copilot-env"


def test_api_key_falls_back_to_github_token() -> None:
    p = object.__new__(CopilotSdkProvider)
    env = {"COPILOT_GITHUB_TOKEN": "", "GITHUB_TOKEN": "gh-token"}
    with patch.dict(os.environ, env, clear=False):
        result = p._verify_and_set_api_key(None)
    assert result == "gh-token"


def test_api_key_returns_none_when_all_absent() -> None:
    """No token at all → None (triggers logged-in CLI user mode)."""
    p = object.__new__(CopilotSdkProvider)
    env = {"COPILOT_GITHUB_TOKEN": "", "GITHUB_TOKEN": "", "GH_TOKEN": ""}
    with patch.dict(os.environ, env, clear=False):
        result = p._verify_and_set_api_key(None)
    assert result is None


# ---------------------------------------------------------------------------
# _init_client
# ---------------------------------------------------------------------------

def test_init_client_stores_token_and_url() -> None:
    p = object.__new__(CopilotSdkProvider)
    p._init_client(api_key="my-token", api_base="localhost:9000")
    assert p._resolved_token == "my-token"
    assert p._cli_url == "localhost:9000"
    assert p._copilot_client is None


def test_init_client_reads_cli_url_from_env() -> None:
    p = object.__new__(CopilotSdkProvider)
    with patch.dict(os.environ, {"COPILOT_CLI_URL": "localhost:7777"}, clear=False):
        p._init_client(api_key=None, api_base=None)
    assert p._cli_url == "localhost:7777"


# ---------------------------------------------------------------------------
# _acompletion (async)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acompletion_returns_chat_completion() -> None:
    provider = _make_provider(api_key="test-token")

    mock_event = MagicMock()
    mock_event.data.content = "The answer is 4."

    mock_session = AsyncMock()
    mock_session.send_and_wait = AsyncMock(return_value=mock_event)
    mock_session.disconnect = AsyncMock()

    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        result = await provider._acompletion(params)

    assert result.choices[0].message.content == "The answer is 4."
    assert result.model == "gpt-4o"
    mock_session.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_acompletion_handles_none_event() -> None:
    """When send_and_wait returns None, content is empty string."""
    provider = _make_provider(api_key="test-token")

    mock_session = AsyncMock()
    mock_session.send_and_wait = AsyncMock(return_value=None)
    mock_session.disconnect = AsyncMock()

    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        result = await provider._acompletion(params)

    assert result.choices[0].message.content == ""


@pytest.mark.asyncio
async def test_acompletion_disconnects_session_on_error() -> None:
    """Session.disconnect() is called even when send_and_wait raises."""
    provider = _make_provider(api_key="test-token")

    mock_session = AsyncMock()
    mock_session.send_and_wait = AsyncMock(side_effect=RuntimeError("CLI died"))
    mock_session.disconnect = AsyncMock()

    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        with pytest.raises(RuntimeError, match="CLI died"):
            await provider._acompletion(params)

    mock_session.disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# _alist_models (async)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_alist_models_converts_model_infos() -> None:
    provider = _make_provider(api_key="test-token")

    raw_models = [
        _make_model_info("gpt-4o", "GPT-4o"),
        _make_model_info("claude-sonnet-4-5", "Claude Sonnet 4.5"),
    ]
    mock_client = AsyncMock()
    mock_client.list_models = AsyncMock(return_value=raw_models)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        models = await provider._alist_models()

    assert len(models) == 2
    ids = {m.id for m in models}
    assert ids == {"gpt-4o", "claude-sonnet-4-5"}
    for m in models:
        assert m.owned_by == "github-copilot"


# ---------------------------------------------------------------------------
# _ensure_client (async)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_client_reuses_existing_instance() -> None:
    """Second call returns the same client without calling start() again."""
    provider = _make_provider(api_key="test-token")

    mock_client = AsyncMock()
    mock_client.start = AsyncMock()
    provider._copilot_client = mock_client  # pre-seed as already initialized

    result = await provider._ensure_client()

    assert result is mock_client
    mock_client.start.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_client_uses_cli_url_path() -> None:
    """When _cli_url is set, only cli_url is passed (no token or cli_path)."""
    provider = _make_provider(api_key="tok", api_base="localhost:9000")

    mock_client = AsyncMock()
    mock_client.start = AsyncMock()

    with patch("any_llm.providers.copilot_sdk.copilot_sdk.CopilotClient", return_value=mock_client) as mock_cls:
        result = await provider._ensure_client()

    assert result is mock_client
    mock_cls.assert_called_once_with({"cli_url": "localhost:9000"})
    mock_client.start.assert_called_once()


# ---------------------------------------------------------------------------
# _messages_to_prompt edge cases
# ---------------------------------------------------------------------------

def test_messages_to_prompt_empty_list() -> None:
    """Empty messages list returns an empty string without raising."""
    assert _messages_to_prompt([]) == ""
