"""Translate Anthropic-style ``cache_control`` breakpoints into Gemini context caches.

A breakpoint is a ``cache_control`` dict (``{"type": "ephemeral"}``, optionally with ``"ttl": "5m"`` or
``"1h"``) on a Chat Completions message, on one of its content parts, or in the message's
``extra_content["cache_control"]`` side-channel that the Messages bridge fills. Gemini caches a single
prefix, so only the last breakpoint counts. A request that uses a cache may not also send
``system_instruction``, ``tools`` or ``tool_config``, so those move into the cache with the contents up
to and including the marked message, and the request sends ``cached_content`` plus the rest.

Caching is best effort: when a cache cannot be created the request is sent as it would have been
without a breakpoint.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from google.genai import errors, types
from pydantic import BaseModel

from any_llm.logging import logger
from any_llm.types.completion import CacheCreationTokenDetails, PromptTokensDetails

from .utils import _convert_messages

if TYPE_CHECKING:
    from google import genai

    from any_llm.types.completion import CompletionUsage

SHORT_TTL_SECONDS = 300
LONG_TTL_SECONDS = 3600
# Entries are dropped this long before Gemini expires the cache, so a request never names a cache
# that expires while it is in flight.
EXPIRY_MARGIN_SECONDS = 15
REGISTRY_MAX_ENTRIES = 1024
# Gemini answers a request naming a missing or expired cache with 403 PERMISSION_DENIED.
_STALE_CACHE_STATUS_CODES = frozenset({403, 404})


@dataclass(frozen=True)
class _RegistryEntry:
    cache_name: str | None
    """``None`` records that Gemini refused to create this cache, typically for being under its minimum size."""
    expires_at: float


class _CacheRegistry:
    """Process-wide LRU map from a cache key to the Gemini cache created for it.

    It is module level because gateways create a provider instance per call, which would leave an
    instance attribute empty on every request.
    """

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[str, _RegistryEntry] = OrderedDict()

    def get(self, key: str, now: float) -> _RegistryEntry | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return entry

    def put(self, key: str, entry: _RegistryEntry) -> None:
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def discard(self, key: str) -> None:
        self._entries.pop(key, None)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


registry = _CacheRegistry(REGISTRY_MAX_ENTRIES)


@dataclass
class ContextCacheUse:
    """The request to send, and how to recover when the cache it names is gone."""

    request_kwargs: dict[str, Any]
    uncached_kwargs: dict[str, Any]
    key: str | None = None
    reused: bool = False
    cache_write_tokens: int | None = None
    ttl_seconds: int = SHORT_TTL_SECONDS

    def is_stale_cache_error(self, exc: Exception) -> bool:
        """Whether ``exc`` means a reused cache no longer exists, so the request can go out uncached."""
        return self.reused and isinstance(exc, errors.ClientError) and exc.code in _STALE_CACHE_STATUS_CODES

    def forget(self) -> None:
        if self.key is not None:
            registry.discard(self.key)


def _breakpoints(message: dict[str, Any]) -> list[dict[str, Any]]:
    marks: list[Any] = [message.get("cache_control")]
    extra_content = message.get("extra_content")
    if isinstance(extra_content, dict):
        marks.append(extra_content.get("cache_control"))
    content = message.get("content")
    if isinstance(content, list):
        marks.extend(part.get("cache_control") for part in content if isinstance(part, dict))
    return [mark for mark in marks if isinstance(mark, dict) and mark.get("type") == "ephemeral"]


def find_last_breakpoint(messages: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Return the index of the last message holding a breakpoint and the TTL of the prefix it ends."""
    last_index: int | None = None
    ttl = SHORT_TTL_SECONDS
    for index, message in enumerate(messages):
        marks = _breakpoints(message)
        if marks:
            last_index = index
            if any(mark.get("ttl") == "1h" for mark in marks):
                ttl = LONG_TTL_SECONDS
    return None if last_index is None else (last_index, ttl)


def _dump(value: Any) -> Any:
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    return value


def client_identity(client: genai.Client) -> dict[str, str | None]:
    """What makes a cache created with ``client`` usable by another client; the API key only as a hash."""
    api_client = client._api_client

    def text(value: Any) -> str | None:
        return value if isinstance(value, str) else None

    api_key = text(api_client.api_key)
    http_options = api_client._http_options
    return {
        "api_key_sha256": hashlib.sha256(api_key.encode()).hexdigest() if api_key else None,
        "base_url": text(http_options.base_url) if http_options is not None else None,
        "project": text(api_client.project),
        "location": text(api_client.location),
    }


def _cache_key(
    provider_name: str,
    identity: dict[str, str | None],
    model: str,
    config: types.GenerateContentConfig,
    prefix: list[types.Content],
) -> str:
    material = {
        "provider": provider_name,
        "identity": identity,
        "model": model,
        "system_instruction": _dump(config.system_instruction),
        "tools": _dump(config.tools),
        "tool_config": _dump(config.tool_config),
        "contents": _dump(prefix),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()


def _prefix_length(
    messages: list[dict[str, Any]], breakpoint_index: int, contents: list[types.Content], provider_name: str
) -> int:
    prefix, _ = _convert_messages(messages[: breakpoint_index + 1], provider_name=provider_name)
    length = len(prefix)
    # Consecutive tool results share one user turn, so a breakpoint inside such a run ends partway
    # through a turn; the cache then stops before that turn.
    if length and (length > len(contents) or contents[length - 1] != prefix[length - 1]):
        length -= 1
    # Gemini rejects a request with no contents, so the last turn always stays outside the cache.
    return min(length, len(contents) - 1)


async def prepare_context_cache(
    client: genai.Client,
    provider_name: str,
    messages: list[dict[str, Any]],
    converted_kwargs: dict[str, Any],
) -> ContextCacheUse:
    """Rewrite ``converted_kwargs`` to use a context cache when ``messages`` carry a breakpoint.

    Reuses a live cache for the same prefix from the registry, otherwise creates one. A caller that set
    ``cached_content`` itself, a request with no breakpoint, and a prefix with nothing to cache are left
    untouched.
    """
    no_cache = ContextCacheUse(request_kwargs=converted_kwargs, uncached_kwargs=converted_kwargs)
    config: types.GenerateContentConfig = converted_kwargs["config"]
    contents: list[types.Content] = converted_kwargs["contents"]
    if config.cached_content is not None or not contents:
        return no_cache
    found = find_last_breakpoint(messages)
    if found is None:
        return no_cache
    breakpoint_index, ttl = found
    length = max(_prefix_length(messages, breakpoint_index, contents, provider_name), 0)
    prefix = contents[:length]
    if not prefix and not config.system_instruction and not config.tools:
        return no_cache

    model: str = converted_kwargs["model"]
    key = _cache_key(provider_name, client_identity(client), model, config, prefix)
    now = time.monotonic()
    entry = registry.get(key, now)
    cache_write_tokens: int | None = None
    if entry is None:
        try:
            cache = await client.aio.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    contents=cast("types.ContentListUnion", prefix) if prefix else None,
                    system_instruction=config.system_instruction,
                    # _convert_tool_spec only produces types.Tool, never the SDK's callables.
                    tools=cast("list[types.Tool] | None", config.tools),
                    tool_config=config.tool_config,
                    ttl=f"{ttl}s",
                ),
            )
        except errors.ClientError as exc:
            logger.debug("Gemini refused a context cache (%s); sending the request uncached", exc.code)
            registry.put(key, _RegistryEntry(cache_name=None, expires_at=now + ttl))
            return no_cache
        except Exception as exc:  # noqa: BLE001
            logger.debug("Creating a Gemini context cache failed (%s); sending the request uncached", exc)
            return no_cache
        if not cache.name:
            return no_cache
        entry = _RegistryEntry(cache_name=cache.name, expires_at=now + ttl - EXPIRY_MARGIN_SECONDS)
        registry.put(key, entry)
        cache_write_tokens = cache.usage_metadata.total_token_count if cache.usage_metadata else None
    if entry.cache_name is None:
        return no_cache

    request_kwargs = {
        **converted_kwargs,
        "config": config.model_copy(
            update={
                "cached_content": entry.cache_name,
                "system_instruction": None,
                "tools": None,
                "tool_config": None,
            }
        ),
        "contents": contents[length:],
    }
    return ContextCacheUse(
        request_kwargs=request_kwargs,
        uncached_kwargs=converted_kwargs,
        key=key,
        reused=cache_write_tokens is None,
        cache_write_tokens=cache_write_tokens,
        ttl_seconds=ttl,
    )


def report_cache_write(usage: CompletionUsage | None, cache_use: ContextCacheUse) -> None:
    """Report the cache this call created as a write rather than a read.

    Gemini counts the prefix as ``cached_content_token_count`` because the request read the cache it
    had just created. ``cached_tokens`` and ``cache_write_tokens`` are disjoint subsets of
    ``prompt_tokens``, so the written tokens leave the read count.
    """
    if usage is None or cache_use.cache_write_tokens is None:
        return
    written = min(cache_use.cache_write_tokens, usage.prompt_tokens)
    details = usage.prompt_tokens_details
    read = details.cached_tokens if details is not None else None
    long_ttl = cache_use.ttl_seconds == LONG_TTL_SECONDS
    usage.prompt_tokens_details = PromptTokensDetails(
        audio_tokens=details.audio_tokens if details is not None else None,
        cached_tokens=max((read or 0) - written, 0),
        cache_write_tokens=written,
        cache_creation_token_details=CacheCreationTokenDetails(
            ephemeral_5m_input_tokens=0 if long_ttl else written,
            ephemeral_1h_input_tokens=written if long_ttl else 0,
        ),
    )
