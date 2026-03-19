"""Unit tests for CopilotSdkProvider."""
from __future__ import annotations

import asyncio
import base64
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.providers.copilot_sdk.copilot_sdk import (
    CopilotSdkProvider,
    _build_chat_completion,
    _build_chunk,
    _cleanup_temp_files,
    _copilot_model_to_openai,
    _extract_attachments,
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
    assert CopilotSdkProvider.SUPPORTS_COMPLETION_STREAMING is True
    assert CopilotSdkProvider.SUPPORTS_COMPLETION_REASONING is True
    assert CopilotSdkProvider.SUPPORTS_COMPLETION_IMAGE is True
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

def _make_session(send_and_wait_result: Any = None, send_and_wait_error: Exception | None = None) -> Any:
    """Build a mock CopilotSession with a sync on() and async send_and_wait/disconnect."""
    session = MagicMock()
    session.on = MagicMock(return_value=lambda: None)  # sync; returns unsubscribe callable
    session.disconnect = AsyncMock()
    if send_and_wait_error:
        session.send_and_wait = AsyncMock(side_effect=send_and_wait_error)
    else:
        session.send_and_wait = AsyncMock(return_value=send_and_wait_result)
    return session


@pytest.mark.asyncio
async def test_acompletion_returns_chat_completion() -> None:
    provider = _make_provider(api_key="test-token")

    mock_event = MagicMock()
    mock_event.data.content = "The answer is 4."

    mock_session = _make_session(send_and_wait_result=mock_event)
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

    mock_session = _make_session(send_and_wait_result=None)
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

    mock_session = _make_session(send_and_wait_error=RuntimeError("CLI died"))
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


# ---------------------------------------------------------------------------
# _build_chat_completion — reasoning field
# ---------------------------------------------------------------------------

def test_build_chat_completion_with_reasoning() -> None:
    cc = _build_chat_completion("Answer", "gpt-4o", reasoning="Because math.")
    assert cc.choices[0].message.reasoning is not None
    assert cc.choices[0].message.reasoning.content == "Because math."


def test_build_chat_completion_no_reasoning() -> None:
    cc = _build_chat_completion("Answer", "gpt-4o")
    assert cc.choices[0].message.reasoning is None


# ---------------------------------------------------------------------------
# _build_chunk
# ---------------------------------------------------------------------------

def test_build_chunk_content_delta() -> None:
    chunk = _build_chunk("hello", "gpt-4o", is_reasoning=False)
    assert chunk.choices[0].delta.content == "hello"
    assert chunk.choices[0].delta.reasoning is None
    assert chunk.object == "chat.completion.chunk"


def test_build_chunk_reasoning_delta() -> None:
    chunk = _build_chunk("step 1", "gpt-4o", is_reasoning=True)
    assert chunk.choices[0].delta.content is None
    assert chunk.choices[0].delta.reasoning is not None
    assert chunk.choices[0].delta.reasoning.content == "step 1"


# ---------------------------------------------------------------------------
# _extract_attachments
# ---------------------------------------------------------------------------

def _make_data_uri(mime: str = "image/jpeg", content: bytes = b"\xff\xd8\xff") -> str:
    return f"data:{mime};base64,{base64.b64encode(content).decode()}"


def test_extract_attachments_base64_image() -> None:
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Look at this"},
        {"type": "image_url", "image_url": {"url": _make_data_uri()}},
    ]}]
    attachments, temp_paths = _extract_attachments(msgs)
    try:
        assert len(attachments) == 1
        assert attachments[0]["type"] == "file"
        assert os.path.exists(attachments[0]["path"])
        assert len(temp_paths) == 1
    finally:
        _cleanup_temp_files(temp_paths)


def test_extract_attachments_http_url_skipped() -> None:
    """HTTP image URLs are not supported (no download) and must be silently skipped."""
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    ]}]
    attachments, temp_paths = _extract_attachments(msgs)
    assert attachments == []
    assert temp_paths == []


def test_extract_attachments_no_images() -> None:
    msgs = [{"role": "user", "content": "Plain text, no images."}]
    attachments, temp_paths = _extract_attachments(msgs)
    assert attachments == []
    assert temp_paths == []


def test_extract_attachments_malformed_data_uri_skipped() -> None:
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,NOT_VALID!!"}},
    ]}]
    # Should not raise; malformed URIs are silently dropped.
    attachments, temp_paths = _extract_attachments(msgs)
    assert attachments == []


def test_cleanup_temp_files_removes_files() -> None:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        path = fh.name
    assert os.path.exists(path)
    _cleanup_temp_files([path])
    assert not os.path.exists(path)


def test_cleanup_temp_files_tolerates_missing() -> None:
    """Cleaning up a path that doesn't exist must not raise."""
    _cleanup_temp_files(["/tmp/does-not-exist-copilot-sdk-test-xyz"])


# ---------------------------------------------------------------------------
# _acompletion — reasoning captured in non-streaming mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acompletion_captures_reasoning() -> None:
    """assistant.reasoning event content is surfaced in ChatCompletion.reasoning."""
    from copilot.generated.session_events import SessionEventType

    provider = _make_provider(api_key="test-token")

    mock_msg_event = MagicMock()
    mock_msg_event.data.content = "42"
    mock_msg_event.type = SessionEventType.ASSISTANT_MESSAGE

    mock_reasoning_event = MagicMock()
    mock_reasoning_event.data.content = "Because 6×7=42."
    mock_reasoning_event.type = SessionEventType.ASSISTANT_REASONING

    def fake_on(callback: Any) -> Any:
        """Immediately fire ASSISTANT_REASONING, then return a no-op unsubscribe."""
        callback(mock_reasoning_event)
        return lambda: None

    mock_session = MagicMock()
    mock_session.on = MagicMock(side_effect=fake_on)
    mock_session.send_and_wait = AsyncMock(return_value=mock_msg_event)
    mock_session.disconnect = AsyncMock()

    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "What is 6×7?"}],
        )
        result = await provider._acompletion(params)

    assert result.choices[0].message.content == "42"
    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content == "Because 6×7=42."


# ---------------------------------------------------------------------------
# _acompletion — reasoning_effort passed to session_cfg
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acompletion_passes_reasoning_effort_to_session() -> None:
    provider = _make_provider(api_key="test-token")

    mock_session = _make_session()
    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Think hard."}],
            reasoning_effort="high",
        )
        await provider._acompletion(params)

    call_cfg = mock_client.create_session.call_args[0][0]
    assert call_cfg.get("reasoning_effort") == "high"


@pytest.mark.asyncio
async def test_acompletion_omits_auto_reasoning_effort() -> None:
    """reasoning_effort='auto' must NOT be forwarded (SDK doesn't accept it)."""
    provider = _make_provider(api_key="test-token")

    mock_session = _make_session()
    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            reasoning_effort="auto",
        )
        await provider._acompletion(params)

    call_cfg = mock_client.create_session.call_args[0][0]
    assert "reasoning_effort" not in call_cfg


# ---------------------------------------------------------------------------
# _acompletion — streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acompletion_streaming_yields_chunks() -> None:
    from copilot.generated.session_events import SessionEventType

    provider = _make_provider(api_key="test-token")

    # Build fake events that the on() callback will receive.
    def _evt(etype: SessionEventType, delta: str) -> MagicMock:
        e = MagicMock()
        e.type = etype
        e.data.delta_content = delta
        return e

    delta_events = [
        _evt(SessionEventType.ASSISTANT_MESSAGE_DELTA, "Hello"),
        _evt(SessionEventType.ASSISTANT_MESSAGE_DELTA, " world"),
        _evt(SessionEventType.SESSION_IDLE, ""),
    ]

    registered_callback: list[Any] = []

    def fake_on(cb: Any) -> Any:
        registered_callback.append(cb)
        return lambda: None

    mock_session = MagicMock()
    mock_session.on = MagicMock(side_effect=fake_on)
    mock_session.disconnect = AsyncMock()

    async def fake_send(opts: Any) -> str:
        for evt in delta_events:
            registered_callback[0](evt)
        return "msg-id"

    mock_session.send = fake_send

    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        stream = await provider._acompletion(params)
        chunks = [c async for c in stream]

    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[1].choices[0].delta.content == " world"
    mock_session.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_acompletion_streaming_reasoning_chunks() -> None:
    from copilot.generated.session_events import SessionEventType

    provider = _make_provider(api_key="test-token")

    def _evt(etype: SessionEventType, delta: str) -> MagicMock:
        e = MagicMock()
        e.type = etype
        e.data.delta_content = delta
        return e

    delta_events = [
        _evt(SessionEventType.ASSISTANT_REASONING_DELTA, "step 1"),
        _evt(SessionEventType.ASSISTANT_MESSAGE_DELTA, "answer"),
        _evt(SessionEventType.SESSION_IDLE, ""),
    ]

    registered_callback: list[Any] = []

    def fake_on(cb: Any) -> Any:
        registered_callback.append(cb)
        return lambda: None

    mock_session = MagicMock()
    mock_session.on = MagicMock(side_effect=fake_on)
    mock_session.disconnect = AsyncMock()

    async def fake_send(opts: Any) -> str:
        for evt in delta_events:
            registered_callback[0](evt)
        return "msg-id"

    mock_session.send = fake_send

    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Think step by step."}],
            stream=True,
        )
        stream = await provider._acompletion(params)
        chunks = [c async for c in stream]

    assert len(chunks) == 2
    # First chunk carries reasoning
    assert chunks[0].choices[0].delta.reasoning is not None
    assert chunks[0].choices[0].delta.reasoning.content == "step 1"
    assert chunks[0].choices[0].delta.content is None
    # Second chunk carries content
    assert chunks[1].choices[0].delta.content == "answer"
    assert chunks[1].choices[0].delta.reasoning is None


@pytest.mark.asyncio
async def test_acompletion_streaming_session_error_raises() -> None:
    """SESSION_ERROR event must raise RuntimeError rather than completing silently."""
    from copilot.generated.session_events import SessionEventType

    provider = _make_provider(api_key="test-token")

    def _evt(etype: SessionEventType, msg: str = "") -> MagicMock:
        e = MagicMock()
        e.type = etype
        e.data.message = msg
        e.data.delta_content = ""
        return e

    delta_events = [
        _evt(SessionEventType.ASSISTANT_MESSAGE_DELTA, ""),  # one content delta first
        _evt(SessionEventType.SESSION_ERROR, "CLI crashed"),
    ]
    # Override delta_content for the first event
    delta_events[0].data.delta_content = "partial"

    registered_callback: list[Any] = []

    def fake_on(cb: Any) -> Any:
        registered_callback.append(cb)
        return lambda: None

    mock_session = MagicMock()
    mock_session.on = MagicMock(side_effect=fake_on)
    mock_session.disconnect = AsyncMock()

    async def fake_send(opts: Any) -> str:
        for evt in delta_events:
            registered_callback[0](evt)
        return "msg-id"

    mock_session.send = fake_send

    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        stream = await provider._acompletion(params)
        with pytest.raises(RuntimeError):
            async for _ in stream:
                pass

    mock_session.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_acompletion_image_attachments_forwarded() -> None:
    """Image attachments extracted from messages must be included in msg_opts sent to session."""
    import base64

    provider = _make_provider(api_key="test-token")

    jpeg_data = base64.b64encode(b"\xff\xd8\xff").decode()
    data_uri = f"data:image/jpeg;base64,{jpeg_data}"

    mock_event = MagicMock()
    mock_event.data.content = "I see an image."
    mock_session = _make_session(send_and_wait_result=mock_event)
    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
        )
        result = await provider._acompletion(params)

    assert result.choices[0].message.content == "I see an image."
    # send_and_wait must have received msg_opts with 'attachments'
    call_kwargs = mock_session.send_and_wait.call_args[0][0]
    assert "attachments" in call_kwargs
    assert len(call_kwargs["attachments"]) == 1
    assert call_kwargs["attachments"][0]["type"] == "file"
