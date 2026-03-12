from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from any_llm.any_llm import AnyLLM

MISSING_PACKAGES_ERROR: ImportError | None = None
try:
    from copilot import CopilotClient, PermissionHandler
    from copilot.types import ModelInfo as CopilotModelInfo
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import Sequence

    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionChunk,
        CompletionParams,
        CreateEmbeddingResponse,
    )
    from any_llm.types.model import Model


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten an OpenAI-style messages list into a single prompt string.

    System messages become an instruction header; prior conversational turns
    are formatted as a transcript; the final user message is the prompt.
    """
    system_parts: list[str] = []
    conversation_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Handle multimodal content blocks — extract text only.
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


def _build_chat_completion(content: str, model_id: str) -> "ChatCompletion":
    """Wrap a plain text response into an OpenAI-compatible ChatCompletion."""
    from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice  # noqa: PLC0415

    return ChatCompletion(
        id=f"copilot-sdk-{int(time.time())}",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                logprobs=None,
            )
        ],
        created=int(time.time()),
        model=model_id,
        object="chat.completion",
    )


def _copilot_model_to_openai(info: "CopilotModelInfo") -> "Model":
    """Convert a copilot-sdk ModelInfo to an OpenAI-compatible Model."""
    from openai.types.model import Model as OpenAIModel  # noqa: PLC0415

    return OpenAIModel(id=info.id, created=0, owned_by="github-copilot", object="model")


class CopilotSdkProvider(AnyLLM):
    """GitHub Copilot SDK provider for any-llm.

    Communicates with the Copilot CLI via JSON-RPC using the ``github-copilot-sdk``
    Python package.  Authentication supports two modes (checked in order):

    1. **Token mode** — set ``COPILOT_GITHUB_TOKEN``, ``GITHUB_TOKEN``, or
       ``GH_TOKEN`` in the environment (or pass ``api_key`` explicitly).
    2. **Logged-in CLI user** — if no token is found, the Copilot CLI uses the
       credentials from the local user's authenticated ``gh`` / ``copilot`` CLI session.

    The Copilot CLI binary must be installed and reachable on ``PATH`` (or pointed
    to via ``COPILOT_CLI_PATH``).  An external CLI server can be used instead by
    setting ``COPILOT_CLI_URL`` (e.g. ``localhost:9000``).

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
    SUPPORTS_COMPLETION_STREAMING = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_IMAGE = False
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
        self._cli_url:        str | None = api_base or os.getenv("COPILOT_CLI_URL")
        self._cli_path:       str | None = os.getenv("COPILOT_CLI_PATH") or None
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

    @override
    async def _acompletion(
        self,
        params: "CompletionParams",
        **kwargs: Any,
    ) -> "ChatCompletion":
        """Send a completion request via a fresh Copilot session."""
        client = await self._ensure_client()
        prompt = _messages_to_prompt(params.messages)
        model_id = params.model_id

        # approve_all silently grants any permissions the session requests
        # (e.g. tool calls).  This mirrors how the Copilot CLI behaves in
        # non-interactive mode and is appropriate for programmatic usage.
        session_cfg: dict[str, Any] = {"on_permission_request": PermissionHandler.approve_all}
        if model_id:
            session_cfg["model"] = model_id

        # A new session is created per request; the Copilot SDK session
        # lifecycle is lightweight (no separate process per session).
        session = await client.create_session(session_cfg)
        try:
            event = await session.send_and_wait({"prompt": prompt})
            content = ""
            if event is not None and hasattr(event, "data") and hasattr(event.data, "content"):
                content = event.data.content or ""
            return _build_chat_completion(content, model_id or self.PROVIDER_NAME)
        finally:
            await session.disconnect()

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
        raise NotImplementedError("CopilotSdkProvider does not support streaming (MVP)")

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
