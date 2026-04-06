from __future__ import annotations

import asyncio
import os
import warnings
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from any_llm.any_llm import AnyLLM

_STREAM_TIMEOUT_SECONDS = 300

MISSING_PACKAGES_ERROR: ImportError | None = None
try:
    from copilot import CopilotClient, PermissionHandler
    from copilot.generated.session_events import SessionEventType

    from .utils import (
        _build_chat_completion,
        _build_chunk,
        _cleanup_temp_files,
        _copilot_model_to_openai,
        _extract_attachments,
        _messages_to_prompt,
    )
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


class CopilotsdkProvider(AnyLLM):
    """GitHub Copilot SDK provider for any-llm.

    Communicates with the Copilot CLI via JSON-RPC using the ``github-copilot-sdk``
    Python package. Authentication supports two modes (checked in order):

    1. **Token mode** — set ``COPILOT_GITHUB_TOKEN`` in the environment (or pass
       ``api_key`` explicitly).
    2. **Logged-in CLI user** — if no token is found, the Copilot CLI uses
       the credentials from the local user's ``gh`` / ``copilot`` CLI session.

    Supports completion (streaming and non-streaming), reasoning
    (``reasoning_effort``), image attachments via ``data:`` URIs, and model
    listing. The binary is bundled by the ``github-copilot-sdk`` wheel;
    ``COPILOT_CLI_URL`` overrides to an external server.

    Environment variables:
        COPILOT_GITHUB_TOKEN: GitHub token with Copilot access (optional).
        COPILOT_CLI_URL: Connect to an external CLI server instead of spawning one.
        COPILOT_CLI_PATH: Override the CLI binary path (default: PATH lookup).
    """

    PROVIDER_NAME = "copilotsdk"
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
    _copilot_client: CopilotClient | None

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        """API key is optional: logged-in CLI mode works without any token."""
        resolved = api_key or os.getenv("COPILOT_GITHUB_TOKEN")
        # Return None (not empty string) so copilot-sdk uses logged-in credentials.
        return resolved or None

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        """Store resolved auth options; actual CLI start is deferred to the first async call."""
        self._resolved_token: str | None = api_key
        self._cli_url: str | None = api_base or os.getenv("COPILOT_CLI_URL")
        self._cli_path: str | None = os.getenv("COPILOT_CLI_PATH") or None
        self._copilot_client = None
        self._client_lock = asyncio.Lock()

    async def _ensure_client(self) -> CopilotClient:
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

    def _build_session_cfg(self, params: CompletionParams, streaming: bool) -> dict[str, Any]:
        """Build a SessionConfig dict from CompletionParams."""
        # PermissionHandler.approve_all silently grants any permissions the session
        # requests (e.g. tool calls). This mirrors how the Copilot CLI behaves in
        # non-interactive mode and is appropriate for programmatic usage. Callers
        # that need a more restrictive policy can subclass and override this method.
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
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Async generator that streams chunks from a live Copilot session.

        Bridges the SDK's event-callback model to an async iterator via an
        asyncio Queue.  The session is disconnected and temp files cleaned up
        when the generator exits (normally or via exception/cancellation).
        """
        queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_event(event: Any) -> None:
            etype = event.type
            if etype == SessionEventType.ASSISTANT_MESSAGE_DELTA:
                loop.call_soon_threadsafe(queue.put_nowait, ("content", event.data.delta_content or ""))
            elif etype == SessionEventType.ASSISTANT_REASONING_DELTA:
                loop.call_soon_threadsafe(queue.put_nowait, ("reasoning", event.data.delta_content or ""))
            elif etype == SessionEventType.SESSION_IDLE:
                loop.call_soon_threadsafe(queue.put_nowait, None)
            elif etype == SessionEventType.SESSION_ERROR:
                # SDK SESSION_ERROR event fields are untyped; use getattr with a
                # default so this path is safe even if the schema changes.
                error_msg = getattr(getattr(event, "data", None), "message", "Copilot session error")
                loop.call_soon_threadsafe(queue.put_nowait, ("error", error_msg))

        unsubscribe = session.on(on_event)
        try:
            await session.send(msg_opts)
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_STREAM_TIMEOUT_SECONDS)
                except TimeoutError:
                    msg = (
                        f"Copilot streaming timed out after {_STREAM_TIMEOUT_SECONDS}s "
                        "waiting for the next event (SESSION_IDLE or SESSION_ERROR never arrived)."
                    )
                    raise RuntimeError(msg) from None
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

    def _warn_unsupported_params(self, params: CompletionParams) -> None:
        """Emit a warning for any CompletionParams fields that this provider cannot honour."""
        unsupported_names = [
            name
            for name, value in {
                "temperature": params.temperature,
                "max_tokens": params.max_tokens,
                "top_p": params.top_p,
                "stop": params.stop,
                "tools": params.tools,
                "tool_choice": params.tool_choice,
            }.items()
            if value is not None
        ]
        if unsupported_names:
            joined = ", ".join(f"'{n}'" for n in unsupported_names)
            warnings.warn(
                f"CopilotsdkProvider does not support {joined} — "
                "these values will be ignored.",
                stacklevel=3,
            )

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Send a completion request via a fresh Copilot session."""
        self._warn_unsupported_params(params)
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
                    # SDK reasoning event fields are untyped; defensive access is
                    # intentional here so a schema change doesn't cause an AttributeError.
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

    async def close(self) -> None:
        """Stop the underlying CopilotClient and release the spawned CLI process."""
        async with self._client_lock:
            if self._copilot_client is not None:
                await self._copilot_client.stop()
                self._copilot_client = None

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        """List models available through the Copilot CLI."""
        client = await self._ensure_client()
        models = await client.list_models()
        return [_copilot_model_to_openai(m) for m in models]

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        msg = "CopilotsdkProvider overrides _acompletion directly"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        msg = "CopilotsdkProvider overrides _acompletion directly"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        msg = "CopilotsdkProvider overrides _acompletion directly"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        msg = "CopilotsdkProvider does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        msg = "CopilotsdkProvider does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        msg = "CopilotsdkProvider uses _alist_models directly"
        raise NotImplementedError(msg)
