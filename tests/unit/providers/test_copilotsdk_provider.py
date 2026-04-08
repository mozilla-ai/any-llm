"""Unit tests for CopilotsdkProvider."""
from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import warnings
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.providers.copilotsdk.copilotsdk import CopilotsdkProvider
from any_llm.providers.copilotsdk.utils import (
    _build_chat_completion,
    _build_chunk,
    _cleanup_temp_files,
    _copilot_model_to_openai,
    _extract_attachments,
    _messages_to_prompt,
)
from any_llm.types.completion import CompletionParams

pytest.importorskip("copilot")


def _make_provider(**kwargs: Any) -> CopilotsdkProvider:
    """Instantiate CopilotsdkProvider while bypassing real CLI startup."""
    provider = object.__new__(CopilotsdkProvider)
    provider._resolved_token = kwargs.get("api_key")
    provider._cli_url = kwargs.get("api_base")
    provider._cli_path = None
    provider._copilot_client = None
    provider._client_lock = asyncio.Lock()
    return provider


def _make_model_info(model_id: str = "gpt-4o", name: str = "GPT-4o") -> Any:
    """Return a minimal mock ModelInfo."""
    model_info = MagicMock()
    model_info.id = model_id
    model_info.name = name
    return model_info


def _make_session(send_and_wait_result: Any = None, send_and_wait_error: Exception | None = None) -> Any:
    """Build a mock Copilot session with a sync on() and async methods."""
    session = MagicMock()
    session.on = MagicMock(return_value=lambda: None)
    session.disconnect = AsyncMock()
    if send_and_wait_error:
        session.send_and_wait = AsyncMock(side_effect=send_and_wait_error)
    else:
        session.send_and_wait = AsyncMock(return_value=send_and_wait_result)
    return session


def _make_data_uri(mime: str = "image/jpeg", content: bytes = b"\xff\xd8\xff") -> str:
    return f"data:{mime};base64,{base64.b64encode(content).decode()}"


def test_messages_to_prompt_simple_user() -> None:
    messages = [{"role": "user", "content": "Hello"}]
    assert _messages_to_prompt(messages) == "User: Hello"


def test_messages_to_prompt_system_prepended() -> None:
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
    ]
    result = _messages_to_prompt(messages)
    assert result.startswith("Be concise.")
    assert "User: Hi" in result


def test_messages_to_prompt_multi_turn() -> None:
    messages = [
        {"role": "user", "content": "Ping"},
        {"role": "assistant", "content": "Pong"},
        {"role": "user", "content": "Again"},
    ]
    result = _messages_to_prompt(messages)
    assert "User: Ping" in result
    assert "Assistant: Pong" in result
    assert "User: Again" in result


def test_messages_to_prompt_multimodal_extracts_text() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "Tell me"}, {"type": "image_url"}]}]
    result = _messages_to_prompt(messages)
    assert "Tell me" in result


def test_messages_to_prompt_empty_list() -> None:
    assert _messages_to_prompt([]) == ""


def test_build_chat_completion_structure() -> None:
    completion = _build_chat_completion("Hello world", "gpt-4o")
    assert completion.choices[0].message.content == "Hello world"
    assert completion.model == "gpt-4o"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.object == "chat.completion"


def test_build_chat_completion_with_reasoning() -> None:
    completion = _build_chat_completion("Answer", "gpt-4o", reasoning="Because math.")
    assert completion.choices[0].message.reasoning is not None
    assert completion.choices[0].message.reasoning.content == "Because math."


def test_build_chat_completion_no_reasoning() -> None:
    completion = _build_chat_completion("Answer", "gpt-4o")
    assert completion.choices[0].message.reasoning is None


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


def test_copilot_model_to_openai_maps_id() -> None:
    info = _make_model_info("claude-sonnet-4-5", "Claude Sonnet 4.5")
    model = _copilot_model_to_openai(info)
    assert model.id == "claude-sonnet-4-5"
    assert model.owned_by == "github-copilot"
    assert model.object == "model"


def test_provider_required_attributes() -> None:
    assert CopilotsdkProvider.PROVIDER_NAME == "copilotsdk"
    assert CopilotsdkProvider.SUPPORTS_COMPLETION is True
    assert CopilotsdkProvider.SUPPORTS_LIST_MODELS is True
    assert CopilotsdkProvider.SUPPORTS_EMBEDDING is False
    assert CopilotsdkProvider.SUPPORTS_COMPLETION_STREAMING is True
    assert CopilotsdkProvider.SUPPORTS_COMPLETION_REASONING is True
    assert CopilotsdkProvider.SUPPORTS_COMPLETION_IMAGE is True
    assert CopilotsdkProvider.MISSING_PACKAGES_ERROR is None


def test_api_key_explicit_wins() -> None:
    provider = object.__new__(CopilotsdkProvider)
    with patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": "env-token"}, clear=False):
        result = provider._verify_and_set_api_key("explicit-token")
    assert result == "explicit-token"


def test_api_key_falls_back_to_copilot_env() -> None:
    provider = object.__new__(CopilotsdkProvider)
    with patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": "copilot-env"}, clear=False):
        result = provider._verify_and_set_api_key(None)
    assert result == "copilot-env"


def test_github_token_env_var_is_not_used() -> None:
    provider = object.__new__(CopilotsdkProvider)
    env = {"COPILOT_GITHUB_TOKEN": "", "GITHUB_TOKEN": "gh-token"}
    with patch.dict(os.environ, env, clear=False):
        result = provider._verify_and_set_api_key(None)
    assert result is None


def test_gh_token_env_var_is_not_used() -> None:
    provider = object.__new__(CopilotsdkProvider)
    env = {"COPILOT_GITHUB_TOKEN": "", "GITHUB_TOKEN": "", "GH_TOKEN": "gh-token"}
    with patch.dict(os.environ, env, clear=False):
        result = provider._verify_and_set_api_key(None)
    assert result is None


def test_api_key_returns_none_when_copilot_token_absent() -> None:
    provider = object.__new__(CopilotsdkProvider)
    env = {"COPILOT_GITHUB_TOKEN": ""}
    with patch.dict(os.environ, env, clear=False):
        result = provider._verify_and_set_api_key(None)
    assert result is None


def test_init_client_stores_token_and_url() -> None:
    provider = object.__new__(CopilotsdkProvider)
    provider._init_client(api_key="my-token", api_base="localhost:9000")
    assert provider._resolved_token == "my-token"  # noqa: S105
    assert provider._cli_url == "localhost:9000"
    assert provider._copilot_client is None
    assert not hasattr(provider, "_extra_kwargs")


def test_init_client_reads_cli_url_from_env() -> None:
    provider = object.__new__(CopilotsdkProvider)
    with patch.dict(os.environ, {"COPILOT_CLI_URL": "localhost:7777"}, clear=False):
        provider._init_client(api_key=None, api_base=None)
    assert provider._cli_url == "localhost:7777"


def test_extract_attachments_base64_image() -> None:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Look at this"},
            {"type": "image_url", "image_url": {"url": _make_data_uri()}},
        ],
    }]
    attachments, temp_paths = _extract_attachments(messages)
    try:
        assert len(attachments) == 1
        assert attachments[0]["type"] == "file"
        assert os.path.exists(attachments[0]["path"])
        assert len(temp_paths) == 1
    finally:
        _cleanup_temp_files(temp_paths)


def test_extract_attachments_http_url_skipped() -> None:
    messages = [{
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}],
    }]
    attachments, temp_paths = _extract_attachments(messages)
    assert attachments == []
    assert temp_paths == []


def test_extract_attachments_no_images() -> None:
    attachments, temp_paths = _extract_attachments([{"role": "user", "content": "Plain text, no images."}])
    assert attachments == []
    assert temp_paths == []


def test_extract_attachments_malformed_data_uri_skipped() -> None:
    messages = [{
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,NOT_VALID!!"}}],
    }]
    attachments, temp_paths = _extract_attachments(messages)
    assert attachments == []
    assert temp_paths == []


def test_cleanup_temp_files_removes_files() -> None:
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        path = handle.name
    assert os.path.exists(path)
    _cleanup_temp_files([path])
    assert not os.path.exists(path)


def test_cleanup_temp_files_tolerates_missing() -> None:
    _cleanup_temp_files(["/tmp/does-not-exist-copilotsdk-test-xyz"])  # noqa: S108


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
    assert {model.id for model in models} == {"gpt-4o", "claude-sonnet-4-5"}
    for model in models:
        assert model.owned_by == "github-copilot"


@pytest.mark.asyncio
async def test_ensure_client_reuses_existing_instance() -> None:
    provider = _make_provider(api_key="test-token")

    mock_client = AsyncMock()
    mock_client.start = AsyncMock()
    provider._copilot_client = mock_client

    result = await provider._ensure_client()

    assert result is mock_client
    mock_client.start.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_client_uses_cli_url_path() -> None:
    provider = _make_provider(api_key="tok", api_base="localhost:9000")

    mock_client = AsyncMock()
    mock_client.start = AsyncMock()

    with patch("any_llm.providers.copilotsdk.copilotsdk.CopilotClient", return_value=mock_client) as mock_cls:
        result = await provider._ensure_client()

    assert result is mock_client
    mock_cls.assert_called_once_with({"cli_url": "localhost:9000"})
    mock_client.start.assert_called_once()


@pytest.mark.asyncio
async def test_acompletion_captures_reasoning() -> None:
    from copilot.generated.session_events import SessionEventType

    provider = _make_provider(api_key="test-token")

    message_event = MagicMock()
    message_event.data.content = "42"
    message_event.type = SessionEventType.ASSISTANT_MESSAGE

    reasoning_event = MagicMock()
    reasoning_event.data.content = "Because 6x7=42."
    reasoning_event.type = SessionEventType.ASSISTANT_REASONING

    def fake_on(callback: Any) -> Any:
        callback(reasoning_event)
        return lambda: None

    mock_session = MagicMock()
    mock_session.on = MagicMock(side_effect=fake_on)
    mock_session.send_and_wait = AsyncMock(return_value=message_event)
    mock_session.disconnect = AsyncMock()

    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "What is 6x7?"}],
        )
        result = await provider._acompletion(params)

    assert result.choices[0].message.content == "42"
    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content == "Because 6x7=42."


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


@pytest.mark.asyncio
async def test_acompletion_streaming_yields_chunks() -> None:
    from copilot.generated.session_events import SessionEventType

    provider = _make_provider(api_key="test-token")

    def make_event(event_type: SessionEventType, delta: str) -> MagicMock:
        event = MagicMock()
        event.type = event_type
        event.data.delta_content = delta
        return event

    delta_events = [
        make_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, "Hello"),
        make_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, " world"),
        make_event(SessionEventType.SESSION_IDLE, ""),
    ]

    registered_callback: list[Any] = []

    def fake_on(callback: Any) -> Any:
        registered_callback.append(callback)
        return lambda: None

    mock_session = MagicMock()
    mock_session.on = MagicMock(side_effect=fake_on)
    mock_session.disconnect = AsyncMock()

    async def fake_send(opts: Any) -> str:
        for event in delta_events:
            registered_callback[0](event)
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
        chunks = [chunk async for chunk in stream]

    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[1].choices[0].delta.content == " world"
    mock_session.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_acompletion_streaming_reasoning_chunks() -> None:
    from copilot.generated.session_events import SessionEventType

    provider = _make_provider(api_key="test-token")

    def make_event(event_type: SessionEventType, delta: str) -> MagicMock:
        event = MagicMock()
        event.type = event_type
        event.data.delta_content = delta
        return event

    delta_events = [
        make_event(SessionEventType.ASSISTANT_REASONING_DELTA, "step 1"),
        make_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, "answer"),
        make_event(SessionEventType.SESSION_IDLE, ""),
    ]

    registered_callback: list[Any] = []

    def fake_on(callback: Any) -> Any:
        registered_callback.append(callback)
        return lambda: None

    mock_session = MagicMock()
    mock_session.on = MagicMock(side_effect=fake_on)
    mock_session.disconnect = AsyncMock()

    async def fake_send(opts: Any) -> str:
        for event in delta_events:
            registered_callback[0](event)
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
        chunks = [chunk async for chunk in stream]

    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.reasoning is not None
    assert chunks[0].choices[0].delta.reasoning.content == "step 1"
    assert chunks[0].choices[0].delta.content is None
    assert chunks[1].choices[0].delta.content == "answer"
    assert chunks[1].choices[0].delta.reasoning is None


@pytest.mark.asyncio
async def test_acompletion_streaming_session_error_raises() -> None:
    from copilot.generated.session_events import SessionEventType

    provider = _make_provider(api_key="test-token")

    def make_event(event_type: SessionEventType, message: str = "") -> MagicMock:
        event = MagicMock()
        event.type = event_type
        event.data.message = message
        event.data.delta_content = ""
        return event

    delta_events = [
        make_event(SessionEventType.ASSISTANT_MESSAGE_DELTA),
        make_event(SessionEventType.SESSION_ERROR, "CLI crashed"),
    ]
    delta_events[0].data.delta_content = "partial"

    registered_callback: list[Any] = []

    def fake_on(callback: Any) -> Any:
        registered_callback.append(callback)
        return lambda: None

    mock_session = MagicMock()
    mock_session.on = MagicMock(side_effect=fake_on)
    mock_session.disconnect = AsyncMock()

    async def fake_send(opts: Any) -> str:
        for event in delta_events:
            registered_callback[0](event)
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
    call_kwargs = mock_session.send_and_wait.call_args[0][0]
    assert "attachments" in call_kwargs
    assert len(call_kwargs["attachments"]) == 1
    assert call_kwargs["attachments"][0]["type"] == "file"


def test_messages_to_prompt_tool_role_warns() -> None:
    messages = [{"role": "tool", "content": "tool result"}]
    with pytest.warns(UserWarning, match="unsupported message role 'tool'"):
        result = _messages_to_prompt(messages)
    assert "tool result" in result


def test_messages_to_prompt_function_role_warns() -> None:
    messages = [{"role": "function", "content": "fn result"}]
    with pytest.warns(UserWarning, match="unsupported message role 'function'"):
        _messages_to_prompt(messages)


def test_extract_attachments_oversized_skipped() -> None:
    from any_llm.providers.copilotsdk import utils as utils_module
    from any_llm.providers.copilotsdk.utils import _MAX_BASE64_DECODE_BYTES

    oversized_bytes = b"\x00" * (_MAX_BASE64_DECODE_BYTES + 1)
    url = f"data:image/jpeg;base64,{base64.b64encode(oversized_bytes).decode()}"
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
    with patch.object(utils_module.logger, "warning") as mock_warning:
        attachments, temp_paths = _extract_attachments(messages)
    assert attachments == []
    assert temp_paths == []
    mock_warning.assert_called_once()
    assert "exceeds" in mock_warning.call_args[0][0]


def test_extract_attachments_malformed_logs_warning() -> None:
    from any_llm.providers.copilotsdk import utils as utils_module

    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,NOT_VALID!!"}}]}]
    with patch.object(utils_module.logger, "warning") as mock_warning:
        attachments, _temp_paths = _extract_attachments(messages)
    assert attachments == []
    mock_warning.assert_called_once()
    assert "failed to decode" in mock_warning.call_args[0][0]


@pytest.mark.asyncio
async def test_acompletion_warns_unsupported_params() -> None:
    provider = _make_provider(api_key="test-token")

    mock_event = MagicMock()
    mock_event.data.content = "ok"
    mock_session = _make_session(send_and_wait_result=mock_event)
    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_tokens=100,
        )
        with pytest.warns(UserWarning, match="'temperature'.*'max_tokens'|'max_tokens'.*'temperature'"):
            await provider._acompletion(params)


@pytest.mark.asyncio
async def test_acompletion_no_warn_for_default_params() -> None:
    provider = _make_provider(api_key="test-token")

    mock_event = MagicMock()
    mock_event.data.content = "ok"
    mock_session = _make_session(send_and_wait_result=mock_event)
    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await provider._acompletion(params)


@pytest.mark.asyncio
async def test_streaming_times_out_when_no_events() -> None:
    provider = _make_provider(api_key="test-token")

    mock_session = MagicMock()
    mock_session.on = MagicMock(return_value=lambda: None)
    mock_session.disconnect = AsyncMock()
    mock_session.send = AsyncMock()

    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        with patch("any_llm.providers.copilotsdk.copilotsdk._STREAM_TIMEOUT_SECONDS", 0.01):
            params = CompletionParams(
                model_id="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )
            stream = await provider._acompletion(params)
            with pytest.raises(RuntimeError, match="timed out"):
                async for _ in stream:
                    pass

    mock_session.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_close_stops_client() -> None:
    provider = _make_provider(api_key="test-token")

    mock_client = AsyncMock()
    mock_client.stop = AsyncMock()
    provider._copilot_client = mock_client

    await provider.close()

    mock_client.stop.assert_called_once()
    assert provider._copilot_client is None


@pytest.mark.asyncio
async def test_close_is_noop_when_no_client() -> None:
    provider = _make_provider(api_key="test-token")
    await provider.close()


@pytest.mark.asyncio
async def test_acompletion_passes_timeout_to_send_and_wait() -> None:
    """Timeout kwarg must be forwarded to send_and_wait, not silently dropped."""
    provider = _make_provider(api_key="test-token")

    mock_event = MagicMock()
    mock_event.data.content = "ok"
    mock_session = _make_session(send_and_wait_result=mock_event)
    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        await provider._acompletion(params, timeout=300.0)

    mock_session.send_and_wait.assert_called_once()
    _opts, actual_timeout = mock_session.send_and_wait.call_args[0][0], mock_session.send_and_wait.call_args[1].get("timeout")
    assert actual_timeout == 300.0


@pytest.mark.asyncio
async def test_acompletion_passes_none_timeout_to_send_and_wait_when_not_provided() -> None:
    """Without a timeout kwarg, None is passed to send_and_wait (SDK uses its own default)."""
    provider = _make_provider(api_key="test-token")

    mock_event = MagicMock()
    mock_event.data.content = "ok"
    mock_session = _make_session(send_and_wait_result=mock_event)
    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)

    with patch.object(provider, "_ensure_client", AsyncMock(return_value=mock_client)):
        params = CompletionParams(
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        await provider._acompletion(params)

    actual_timeout = mock_session.send_and_wait.call_args[1].get("timeout")
    assert actual_timeout is None


@pytest.mark.asyncio
async def test_streaming_uses_custom_timeout_from_kwargs() -> None:
    """A timeout kwarg must be used as the per-event wait in the streaming path."""
    from copilot.generated.session_events import SessionEventType

    provider = _make_provider(api_key="test-token")

    registered_callback: list[Any] = []

    def fake_on(callback: Any) -> Any:
        registered_callback.append(callback)
        return lambda: None

    mock_session = MagicMock()
    mock_session.on = MagicMock(side_effect=fake_on)
    mock_session.disconnect = AsyncMock()

    async def fake_send(opts: Any) -> str:
        # Do not fire any events so the stream blocks until the custom 0.01s timeout fires.
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
        stream = await provider._acompletion(params, timeout=0.01)
        with pytest.raises(RuntimeError, match="timed out after 0.01s"):
            async for _ in stream:
                pass

    mock_session.disconnect.assert_called_once()
