from __future__ import annotations

import base64
import logging
import mimetypes
import os
import tempfile
import time
import warnings
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Eagerly initialise the MIME-type database so that mimetypes.guess_extension()
# is thread-safe from the very first call (lazy initialisation is not thread-safe).
mimetypes.init()

_MAX_BASE64_DECODE_BYTES = 20 * 1024 * 1024  # 20 MiB

if TYPE_CHECKING:
    from copilot.types import ModelInfo as CopilotModelInfo

    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionChunk,
    )
    from any_llm.types.model import Model


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
        elif role == "user":
            conversation_parts.append(f"User: {content}")
        else:
            warnings.warn(
                f"CopilotsdkProvider: unsupported message role '{role}' — rendered as User turn.",
                stacklevel=2,
            )
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
                if len(raw) > _MAX_BASE64_DECODE_BYTES:
                    logger.warning(
                        "CopilotsdkProvider: skipping image attachment — decoded size exceeds %d MiB limit",
                        _MAX_BASE64_DECODE_BYTES // (1024 * 1024),
                    )
                    continue
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
                    fh.write(raw)
                    temp_paths.append(fh.name)
                    attachments.append({"type": "file", "path": fh.name})
            except Exception:
                logger.warning("CopilotsdkProvider: failed to decode image attachment — skipping.", exc_info=True)

    return attachments, temp_paths


def _cleanup_temp_files(paths: list[str]) -> None:
    """Remove temporary image files created by :func:`_extract_attachments`."""
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def _build_chat_completion(
    content: str,
    model_id: str,
    reasoning: str | None = None,
) -> ChatCompletion:
    """Wrap a plain text response into an OpenAI-compatible ChatCompletion."""
    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionMessage,
        Choice,
        Reasoning,
    )

    return ChatCompletion(
        id=f"copilotsdk-{time.time_ns()}",
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
        # The Copilot SDK does not expose token counts, so usage is omitted.
        # Gateway budget tracking will not function for this provider.
    )


def _build_chunk(delta: str, model_id: str, *, is_reasoning: bool = False) -> ChatCompletionChunk:
    """Wrap a streaming delta into an OpenAI-compatible ChatCompletionChunk."""
    from any_llm.types.completion import (
        ChatCompletionChunk,
        ChoiceDelta,
        ChunkChoice,
        Reasoning,
    )

    return ChatCompletionChunk(
        id=f"copilotsdk-{time.time_ns()}",
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


def _copilot_model_to_openai(info: CopilotModelInfo) -> Model:
    """Convert a copilot-sdk ModelInfo to an OpenAI-compatible Model."""
    from openai.types.model import Model as OpenAIModel

    return OpenAIModel(id=info.id, created=0, owned_by="github-copilot", object="model")
