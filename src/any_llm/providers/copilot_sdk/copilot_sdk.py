from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import tempfile
import time
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from any_llm.any_llm import AnyLLM

# Eagerly initialise the MIME-type database so that mimetypes.guess_extension()
# is thread-safe from the very first call (lazy initialisation is not thread-safe).
mimetypes.init()

MISSING_PACKAGES_ERROR: ImportError | None = None
try:
    from copilot import CopilotClient, PermissionHandler
    from copilot.generated.session_events import SessionEventType
    from copilot.types import ModelInfo as CopilotModelInfo
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionChunk,
        CompletionParams,
        CreateEmbeddingResponse,
    )
    from any_llm.types.model import Model


# ---------------------------------------------------------------------------
# Message / prompt helpers
# ---------------------------------------------------------------------------

def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten an OpenAI-style messages list into a single prompt string.

    System messages become an instruction header; prior conversational turns
    are formatted as a transcript; the final user message is the prompt.
    Image content blocks are intentionally omitted here — they are forwarded
    as file attachments via :func:`_extract_attachments`.
    """
    system_parts: list[str] = []
    conversation_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Multimodal content blocks: extract text only; images handled separately.
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        if role == "system":
            system_parts.append(str(content))
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {content}")
        else:
            conversation_parts.append(f"User: {content}")

    parts: list[str] = []
    if system_parts:
        parts.append("\n".join(system_parts))
    if conversation_parts:
        parts.append("\n\n".join(conversation_parts))
    return "\n\n".join(parts)


def _extract_attachments(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract ``image_url`` content blocks from messages as file attachments.

    Decodes base64 ``data:`` URIs to temporary files on disk, which the
    Copilot SDK then passes to the CLI as ``FileAttachment`` objects.

    Returns:
        (attachments, temp_paths) where ``temp_paths`` must be cleaned up
        by the caller after the ``session.send()`` call completes.
    """
    attachments: list[dict[str, Any]] = []
    temp_paths: list[str] = []

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            url = (block.get("image_url") or {}).get("url", "")
            if not url.startswith("data:"):
                # HTTP/HTTPS URLs would need downloading — skipped for now.
                continue
            try:
                header, b64data = url.split(",", 1)
                mime = header.split(";")[0].split(":")[1]
                ext = mimetypes.guess_extension(mime) or ".bin"
                # guess_extension returns ".jpe" for image/jpeg on some platforms.
                if ext == ".jpe":
                    ext = ".jpg"
                raw = base64.b64decode(b64data, validate=True)
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
                    fh.write(raw)
                    temp_paths.append(fh.name)
                    attachments.append({"type": "file", "path": fh.name})
            except Exception:  # noqa: BLE001
                pass  # Skip malformed data URIs rather than failing the whole request.

    return attachments, temp_paths


def _cleanup_temp_files(paths: list[str]) -> None:
    """Remove temporary image files created by :func:`_extract_attachments`."""
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Response / chunk builders
# ---------------------------------------------------------------------------

def _build_chat_completion(
    content: str,
    model_id: str,
    reasoning: str | None = None,
) -> "ChatCompletion":
    """Wrap a plain text response into an OpenAI-compatible ChatCompletion."""
    from any_llm.types.completion import (  # noqa: PLC0415
        ChatCompletion,
        ChatCompletionMessage,
        Choice,
        Reasoning,
    )

    return ChatCompletion(
        id=f"copilot-sdk-{int(time.time())}",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                    reasoning=Reasoning(content=reasoning) if reasoning else None,
                ),
                logprobs=None,
            )
        ],
        created=int(time.time()),
        model=model_id,
        object="chat.completion",
    )


def _build_chunk(delta: str, model_id: str, *, is_reasoning: bool = False) -> "ChatCompletionChunk":
    """Wrap a streaming delta into an OpenAI-compatible ChatCompletionChunk."""
    from any_llm.types.completion import (  # noqa: PLC0415
        ChatCompletionChunk,
        ChunkChoice,
        ChoiceDelta,
        Reasoning,
    )

    return ChatCompletionChunk(
        id=f"copilot-sdk-{time.time_ns()}",
        choices=[
            ChunkChoice(
                delta=ChoiceDelta(
                    role="assistant",
                    content=None if is_reasoning else delta,
                    reasoning=Reasoning(content=delta) if is_reasoning else None,
                ),
                finish_reason=None,
                index=0,
            )
        ],
        created=int(time.time()),
        model=model_id,
        object="chat.completion.chunk",
    )


def _copilot_model_to_openai(info: "CopilotModelInfo") -> "Model":
    """Convert a copilot-sdk ModelInfo to an OpenAI-compatible Model."""
    from openai.types.model import Model as OpenAIModel  # noqa: PLC0415

    return OpenAIModel(id=info.id, created=0, owned_by="github-copilot", object="model")


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class CopilotSdkProvider(AnyLLM):
    """GitHub Copilot SDK provider for any-llm.

    Communicates with the Copilot CLI via JSON-RPC using the ``github-copilot-sdk``
    Python package.  Authentication supports two modes (checked in order):

    1. **Token mode** — set ``COPILOT_GITHUB_TOKEN``, ``GITHUB_TOKEN``, or
       ``GH_TOKEN`` in the environment (or pass ``api_key`` explicitly).
    2. **Logged-in CLI user** — if no token is found, the Copilot CLI uses the
       credentials from the local user's authenticated ``gh`` / ``copilot`` CLI session.

    The Copilot CLI binary must be installed and reachable on ``PATH`` (or pointed
    to via ``COPILOT_CLI_PATH``).  The platform-specific ``github-copilot-sdk``
    wheel (e.g. ``github-copilot-sdk==X.Y.Z`` targeting your OS/arch) bundles the
    binary automatically.  An external CLI server can be used instead by setting
    ``COPILOT_CLI_URL`` (e.g. ``localhost:9000``).

    **Supported features:**

    * Completion (non-streaming and streaming)
    * Reasoning (``reasoning_effort``: ``low`` / ``medium`` / ``high`` / ``xhigh``)
    * Image attachments via ``image_url`` content blocks (``data:`` URIs only)
    * Model listing

    Environment variables:
        COPILOT_GITHUB_TOKEN: GitHub token with Copilot access (optional).
        GITHUB_TOKEN / GH_TOKEN: Fallback GitHub token sources (optional).
        COPILOT_CLI_URL: Connect to an external CLI server instead of spawning one.
        COPILOT_CLI_PATH: Override the CLI binary path (default: PATH lookup).
    """

    PROVIDER_NAME = "copilot_sdk"
    ENV_API_KEY_NAME = "COPILOT_GITHUB_TOKEN"
    ENV_API_BASE_NAME = "COPILOT_CLI_URL"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/github/copilot-sdk"

    SUPPORTS_COMPLETION = True
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_RESPONSES = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    # Internal state — populated lazily on first call.
    _copilot_client: "CopilotClient | None"

    # ------------------------------------------------------------------ auth --

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        """API key is optional: logged-in CLI mode works without any token."""
        resolved = (
            api_key
            or os.getenv("COPILOT_GITHUB_TOKEN")
            or os.getenv("GITHUB_TOKEN")
            or os.getenv("GH_TOKEN")
        )
        # Return None (not empty string) so copilot-sdk uses logged-in credentials.
        return resolved or None

    # ---------------------------------------------------------- client init --

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        """Store resolved auth options; actual CLI start is deferred to the first async call."""
        self._resolved_token: str | None = api_key
        self._cli_url: str | None = api_base or os.getenv("COPILOT_CLI_URL")
        self._cli_path: str | None = os.getenv("COPILOT_CLI_PATH") or None
        self._extra_kwargs = kwargs
        self._copilot_client = None
        self._client_lock = asyncio.Lock()

    async def _ensure_client(self) -> "CopilotClient":
        """Lazily create and start a CopilotClient, reusing it across calls.

        The lock prevents concurrent coroutines from each spawning a separate
        CLI process when the client has not yet been initialized.
        """
        if self._copilot_client is not None:
            return self._copilot_client

        async with self._client_lock:
            # Re-check inside the lock: another coroutine may have initialized
            # the client while we were waiting.
            if self._copilot_client is not None:
                return self._copilot_client

            opts: dict[str, Any] = {}
            if self._cli_url:
                opts["cli_url"] = self._cli_url
            else:
                if self._cli_path:
                    opts["cli_path"] = self._cli_path
                if self._resolved_token:
                    opts["github_token"] = self._resolved_token

            # Pass None (not {}) so the SDK uses its own defaults when no
            # options are configured.
            self._copilot_client = CopilotClient(opts or None)
            await self._copilot_client.start()

        return self._copilot_client

    # --------------------------------------------------- completion (async) --

    def _build_session_cfg(self, params: "CompletionParams", streaming: bool) -> dict[str, Any]:
        """Build a SessionConfig dict from CompletionParams."""
        # approve_all silently grants any permissions the session requests
        # (e.g. tool calls).  This mirrors how the Copilot CLI behaves in
        # non-interactive mode and is appropriate for programmatic usage.
        cfg: dict[str, Any] = {"on_permission_request": PermissionHandler.approve_all}
        if params.model_id:
            cfg["model"] = params.model_id
        # Pass reasoning_effort through; omit values the SDK doesn't accept.
        if params.reasoning_effort and params.reasoning_effort not in ("auto", "none"):
            cfg["reasoning_effort"] = params.reasoning_effort
        if streaming:
            cfg["streaming"] = True
        return cfg

    async def _stream_from_session(
        self,
        session: Any,
        msg_opts: dict[str, Any],
        model_id: str,
        temp_paths: list[str],
    ) -> "AsyncIterator[ChatCompletionChunk]":
        """Async generator that streams chunks from a live Copilot session.

        Bridges the SDK's event-callback model to an async iterator via an
        asyncio Queue.  The session is disconnected and temp files cleaned up
        when the generator exits (normally or via exception/cancellation).
        """
        queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()

        def on_event(event: Any) -> None:
            etype = event.type
            if etype == SessionEventType.ASSISTANT_MESSAGE_DELTA:
                queue.put_nowait(("content", event.data.delta_content or ""))
            elif etype == SessionEventType.ASSISTANT_REASONING_DELTA:
                queue.put_nowait(("reasoning", event.data.delta_content or ""))
            elif etype == SessionEventType.SESSION_IDLE:
                queue.put_nowait(None)  # sentinel — streaming complete normally
            elif etype == SessionEventType.SESSION_ERROR:
                queue.put_nowait(("error", getattr(getattr(event, "data", None), "message", "Copilot session error")))

        unsubscribe = session.on(on_event)
        try:
            await session.send(msg_opts)
            while True:
                item = await queue.get()
                if item is None:
                    break
                kind, delta = item
                if kind == "error":
                    raise RuntimeError(delta)
                yield _build_chunk(delta, model_id, is_reasoning=(kind == "reasoning"))
        finally:
            unsubscribe()
            await session.disconnect()
            _cleanup_temp_files(temp_paths)

    @override
    async def _acompletion(
        self,
        params: "CompletionParams",
        **kwargs: Any,
    ) -> "ChatCompletion | AsyncIterator[ChatCompletionChunk]":
        """Send a completion request via a fresh Copilot session."""
        client = await self._ensure_client()
        prompt = _messages_to_prompt(params.messages)
        model_id = params.model_id or self.PROVIDER_NAME
        attachments, temp_paths = _extract_attachments(params.messages)

        session_cfg = self._build_session_cfg(params, streaming=bool(params.stream))
        # A new session is created per request; the Copilot SDK session
        # lifecycle is lightweight (no separate process per session).
        session = await client.create_session(session_cfg)

        msg_opts: dict[str, Any] = {"prompt": prompt}
        if attachments:
            msg_opts["attachments"] = attachments

        if params.stream:
            # Return the async generator directly; it owns the session lifecycle.
            return self._stream_from_session(session, msg_opts, model_id, temp_paths)

        # Non-streaming: capture reasoning alongside the final message event.
        try:
            reasoning_content: str | None = None

            def on_reasoning(event: Any) -> None:
                nonlocal reasoning_content
                if event.type == SessionEventType.ASSISTANT_REASONING:
                    reasoning_content = (
                        getattr(getattr(event, "data", None), "content", None) or None
                    )

            unsubscribe = session.on(on_reasoning)
            try:
                event = await session.send_and_wait(msg_opts)
            finally:
                unsubscribe()

            content = ""
            if event is not None and hasattr(event, "data") and hasattr(event.data, "content"):
                content = event.data.content or ""
            return _build_chat_completion(content, model_id, reasoning_content)
        finally:
            await session.disconnect()
            _cleanup_temp_files(temp_paths)

    # -------------------------------------------------- model listing (async) --

    @override
    async def _alist_models(self, **kwargs: Any) -> "Sequence[Model]":
        """List models available through the Copilot CLI."""
        client = await self._ensure_client()
        models = await client.list_models()
        return [_copilot_model_to_openai(m) for m in models]

    # ------------ Required abstract stubs (unused — _acompletion overridden) --

    @staticmethod
    @override
    def _convert_completion_params(params: "CompletionParams", **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("CopilotSdkProvider overrides _acompletion directly")

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> "ChatCompletion":
        raise NotImplementedError("CopilotSdkProvider overrides _acompletion directly")

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> "ChatCompletionChunk":
        raise NotImplementedError("CopilotSdkProvider overrides _acompletion directly")

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("CopilotSdkProvider does not support embeddings")

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> "CreateEmbeddingResponse":
        raise NotImplementedError("CopilotSdkProvider does not support embeddings")

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> "Sequence[Model]":
        raise NotImplementedError("CopilotSdkProvider uses _alist_models directly")
