from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors, types

from any_llm.providers.gemini import GeminiProvider
from any_llm.providers.gemini import context_cache
from any_llm.providers.gemini.context_cache import (
    EXPIRY_MARGIN_SECONDS,
    ContextCacheUse,
    _CacheRegistry,
    _RegistryEntry,
    client_identity,
    find_last_breakpoint,
    registry,
    report_cache_write,
)
from any_llm.types.completion import ChatCompletion, CompletionParams, CompletionUsage, PromptTokensDetails
from any_llm.types.messages import MessageResponse

MODEL = "gemini-2.5-flash"
CACHE_NAME = "cachedContents/abc"
CACHED_TOKENS = 2000
PROMPT_TOKENS = 2010
SYSTEM = "You answer from the manual."
EPHEMERAL = {"type": "ephemeral"}
WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
}


@pytest.fixture(autouse=True)
def _empty_registry() -> Iterator[None]:
    registry.clear()
    yield
    registry.clear()


def _response(cached_tokens: int | None = CACHED_TOKENS) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="42")]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=PROMPT_TOKENS,
            cached_content_token_count=cached_tokens,
            candidates_token_count=5,
            total_token_count=PROMPT_TOKENS + 5,
        ),
        model_version=MODEL,
    )


def _cache(total_token_count: int | None = CACHED_TOKENS, name: str | None = CACHE_NAME) -> types.CachedContent:
    usage = types.CachedContentUsageMetadata(total_token_count=total_token_count) if total_token_count else None
    return types.CachedContent(name=name, usage_metadata=usage)


def _client(api_key: str = "key-a") -> MagicMock:
    client = MagicMock()
    client._api_client.api_key = api_key
    client._api_client.project = None
    client._api_client.location = None
    client._api_client._http_options.base_url = None
    client.aio.caches.create = AsyncMock(return_value=_cache())
    client.aio.models.generate_content = AsyncMock(return_value=_response())
    return client


def _provider(client: MagicMock) -> GeminiProvider:
    with patch("any_llm.providers.gemini.gemini.genai.Client", return_value=client):
        return GeminiProvider(api_key=client._api_client.api_key)


async def _complete(client: MagicMock, messages: list[dict[str, Any]], **kwargs: Any) -> ChatCompletion:
    params = CompletionParams(model_id=MODEL, messages=messages, **kwargs.pop("params", {}))
    result = await _provider(client)._acompletion(params, **kwargs)
    assert isinstance(result, ChatCompletion)
    return result


def _cache_config(client: MagicMock) -> types.CreateCachedContentConfig:
    config: types.CreateCachedContentConfig = client.aio.caches.create.call_args.kwargs["config"]
    return config


def _sent(client: MagicMock) -> dict[str, Any]:
    sent: dict[str, Any] = client.aio.models.generate_content.call_args.kwargs
    return sent


def _texts(contents: list[types.Content] | None) -> list[str | None]:
    return [part.text for content in contents or [] for part in content.parts or []]


@pytest.mark.asyncio
async def test_breakpoint_on_system_message_moves_system_and_tools_into_the_cache() -> None:
    client = _client()
    messages = [{"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}, {"role": "user", "content": "Q"}]

    await _complete(client, messages, params={"tools": [WEATHER_TOOL], "tool_choice": "auto"})

    cache_config = _cache_config(client)
    assert client.aio.caches.create.call_args.kwargs["model"] == MODEL
    assert cache_config.system_instruction == SYSTEM
    assert cache_config.tools
    assert cache_config.tool_config is not None
    assert cache_config.contents is None
    assert cache_config.ttl == "300s"
    sent = _sent(client)
    assert sent["config"].cached_content == CACHE_NAME
    assert sent["config"].system_instruction is None
    assert sent["config"].tools is None
    assert sent["config"].tool_config is None
    assert _texts(sent["contents"]) == ["Q"]


@pytest.mark.asyncio
async def test_breakpoint_on_a_content_part_caches_contents_up_to_its_message() -> None:
    client = _client()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Doc", "cache_control": EPHEMERAL}]},
        {"role": "assistant", "content": "Read it."},
        {"role": "user", "content": "Q"},
    ]

    await _complete(client, messages)

    cache_config = _cache_config(client)
    assert isinstance(cache_config.contents, list)
    assert _texts(cache_config.contents) == ["Doc"]
    assert _texts(_sent(client)["contents"]) == ["Read it.", "Q"]


@pytest.mark.asyncio
async def test_breakpoint_in_extra_content_side_channel_is_honored() -> None:
    client = _client()
    messages = [
        {"role": "system", "content": SYSTEM, "extra_content": {"cache_control": EPHEMERAL}},
        {"role": "user", "content": "Q"},
    ]

    await _complete(client, messages)

    assert _cache_config(client).system_instruction == SYSTEM
    assert _sent(client)["config"].cached_content == CACHE_NAME


@pytest.mark.asyncio
async def test_last_breakpoint_wins_and_a_1h_breakpoint_in_the_prefix_sets_the_ttl() -> None:
    client = _client()
    messages = [
        {"role": "system", "content": SYSTEM, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"role": "user", "content": "Doc", "cache_control": EPHEMERAL},
        {"role": "assistant", "content": "Read it."},
        {"role": "user", "content": "Q"},
    ]

    await _complete(client, messages)

    cache_config = _cache_config(client)
    assert isinstance(cache_config.contents, list)
    assert _texts(cache_config.contents) == ["Doc"]
    assert cache_config.ttl == "3600s"


@pytest.mark.asyncio
async def test_breakpoint_on_the_last_message_keeps_that_turn_outside_the_cache() -> None:
    client = _client()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Q", "cache_control": EPHEMERAL}]

    await _complete(client, messages)

    assert _cache_config(client).contents is None
    assert _cache_config(client).system_instruction == SYSTEM
    assert _texts(_sent(client)["contents"]) == ["Q"]


@pytest.mark.asyncio
async def test_breakpoint_inside_a_run_of_tool_results_stops_before_their_shared_turn() -> None:
    client = _client()
    calls = [
        {"id": f"call_{i}", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
        for i in (1, 2)
    ]
    messages = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": None, "tool_calls": calls},
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny", "cache_control": EPHEMERAL},
        {"role": "tool", "tool_call_id": "call_2", "content": "Rainy"},
    ]

    await _complete(client, messages)

    cache_config = _cache_config(client)
    assert isinstance(cache_config.contents, list)
    assert [content.role for content in cache_config.contents] == ["user", "model"]
    sent_contents = _sent(client)["contents"]
    assert len(sent_contents) == 1
    assert len(sent_contents[0].parts) == 2


@pytest.mark.asyncio
async def test_nothing_to_cache_sends_the_request_untouched() -> None:
    client = _client()

    await _complete(client, [{"role": "user", "content": "Q", "cache_control": EPHEMERAL}])

    client.aio.caches.create.assert_not_called()
    assert _sent(client)["config"].cached_content is None


@pytest.mark.asyncio
async def test_no_breakpoint_makes_no_caches_call() -> None:
    client = _client()

    await _complete(client, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Q"}])

    client.aio.caches.create.assert_not_called()
    assert _sent(client)["config"].system_instruction == SYSTEM
    assert _sent(client)["config"].cached_content is None


@pytest.mark.asyncio
async def test_explicit_cached_content_bypasses_the_automatic_path() -> None:
    client = _client()
    messages = [{"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}, {"role": "user", "content": "Q"}]

    result = await _complete(client, messages, cached_content="cachedContents/mine")

    client.aio.caches.create.assert_not_called()
    assert _sent(client)["config"].cached_content == "cachedContents/mine"
    assert result.usage is not None
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == CACHED_TOKENS
    assert result.usage.prompt_tokens_details.cache_write_tokens is None


@pytest.mark.asyncio
async def test_created_cache_is_reported_as_a_write_not_a_read() -> None:
    client = _client()
    messages = [{"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}, {"role": "user", "content": "Q"}]

    result = await _complete(client, messages)

    assert result.usage is not None
    assert result.usage.prompt_tokens == PROMPT_TOKENS
    details = result.usage.prompt_tokens_details
    assert details is not None
    assert details.cache_write_tokens == CACHED_TOKENS
    assert details.cached_tokens == 0
    assert details.cache_creation_token_details is not None
    assert details.cache_creation_token_details.ephemeral_5m_input_tokens == CACHED_TOKENS
    assert details.cache_creation_token_details.ephemeral_1h_input_tokens == 0


@pytest.mark.asyncio
async def test_cache_is_reused_across_provider_instances() -> None:
    first_client, second_client = _client(), _client()
    messages = [{"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}, {"role": "user", "content": "Q"}]

    await _complete(first_client, messages)
    result = await _complete(second_client, messages)

    second_client.aio.caches.create.assert_not_called()
    assert _sent(second_client)["config"].cached_content == CACHE_NAME
    assert result.usage is not None
    details = result.usage.prompt_tokens_details
    assert details is not None
    assert details.cached_tokens == CACHED_TOKENS
    assert details.cache_write_tokens is None


@pytest.mark.asyncio
async def test_different_api_keys_do_not_share_caches() -> None:
    first_client, second_client = _client("key-a"), _client("key-b")
    messages = [{"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}, {"role": "user", "content": "Q"}]

    await _complete(first_client, messages)
    await _complete(second_client, messages)

    first_client.aio.caches.create.assert_awaited_once()
    second_client.aio.caches.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_is_recreated_once_its_entry_nears_expiry() -> None:
    client = _client()
    messages = [{"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}, {"role": "user", "content": "Q"}]

    with patch.object(context_cache, "time") as clock:
        clock.monotonic.return_value = 1000.0
        await _complete(client, messages)
        clock.monotonic.return_value = 1000.0 + 300 - EXPIRY_MARGIN_SECONDS - 1
        await _complete(client, messages)
        assert client.aio.caches.create.await_count == 1
        clock.monotonic.return_value = 1000.0 + 300 - EXPIRY_MARGIN_SECONDS
        await _complete(client, messages)

    assert client.aio.caches.create.await_count == 2


@pytest.mark.asyncio
async def test_refused_cache_falls_back_and_is_negative_cached() -> None:
    client = _client()
    client.aio.caches.create.side_effect = errors.ClientError(
        400, {"error": {"code": 400, "message": "Cached content is too small.", "status": "INVALID_ARGUMENT"}}
    )
    messages = [{"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}, {"role": "user", "content": "Q"}]

    first = await _complete(client, messages)
    await _complete(client, messages)

    client.aio.caches.create.assert_awaited_once()
    sent = _sent(client)
    assert sent["config"].cached_content is None
    assert sent["config"].system_instruction == SYSTEM
    assert first.usage is not None
    assert first.usage.prompt_tokens_details is not None
    assert first.usage.prompt_tokens_details.cache_write_tokens is None


@pytest.mark.asyncio
async def test_transient_creation_failure_falls_back_without_negative_caching() -> None:
    client = _client()
    client.aio.caches.create.side_effect = errors.ServerError(503, {"error": {"code": 503, "message": "busy"}})
    messages = [{"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}, {"role": "user", "content": "Q"}]

    await _complete(client, messages)
    await _complete(client, messages)

    assert client.aio.caches.create.await_count == 2
    assert _sent(client)["config"].system_instruction == SYSTEM


@pytest.mark.asyncio
async def test_cache_created_without_a_name_falls_back() -> None:
    client = _client()
    client.aio.caches.create.return_value = _cache(name=None)
    messages = [{"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}, {"role": "user", "content": "Q"}]

    await _complete(client, messages)

    assert _sent(client)["config"].cached_content is None
    assert len(registry) == 0


@pytest.mark.asyncio
async def test_reused_cache_that_is_gone_is_forgotten_and_the_request_resent_uncached() -> None:
    client = _client()
    messages = [{"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}, {"role": "user", "content": "Q"}]
    await _complete(client, messages)
    missing = errors.ClientError(403, {"error": {"code": 403, "message": "CachedContent not found"}})
    client.aio.models.generate_content.side_effect = [missing, _response(cached_tokens=None)]

    await _complete(client, messages)

    retried = _sent(client)
    assert retried["config"].cached_content is None
    assert retried["config"].system_instruction == SYSTEM
    assert len(registry) == 0


@pytest.mark.asyncio
async def test_other_errors_on_a_reused_cache_propagate() -> None:
    client = _client()
    messages = [{"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}, {"role": "user", "content": "Q"}]
    await _complete(client, messages)
    client.aio.models.generate_content.side_effect = errors.ClientError(400, {"error": {"code": 400, "message": "bad"}})

    with pytest.raises(errors.ClientError):
        await _complete(client, messages)

    assert len(registry) == 1


async def _aiter(items: list[types.GenerateContentResponse]) -> AsyncIterator[types.GenerateContentResponse]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_streamed_completion_reports_the_cache_write_on_usage_chunks() -> None:
    client = _client()
    client.aio.models.generate_content_stream = AsyncMock(return_value=_aiter([_response()]))
    messages = [{"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}, {"role": "user", "content": "Q"}]

    stream = await _provider(client)._acompletion(CompletionParams(model_id=MODEL, messages=messages, stream=True))

    assert not isinstance(stream, ChatCompletion)
    chunks = [chunk async for chunk in stream]
    assert client.aio.models.generate_content_stream.call_args.kwargs["config"].cached_content == CACHE_NAME
    usage = chunks[-1].usage
    assert usage is not None
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cache_write_tokens == CACHED_TOKENS
    assert usage.prompt_tokens_details.cached_tokens == 0


@pytest.mark.asyncio
async def test_messages_bridge_reports_creation_then_read() -> None:
    first_client, second_client = _client(), _client()
    system = [{"type": "text", "text": SYSTEM, "cache_control": EPHEMERAL}]
    messages = [{"role": "user", "content": "Q"}]

    first = await _provider(first_client).amessages(model=MODEL, system=system, messages=messages, max_tokens=64)
    second = await _provider(second_client).amessages(model=MODEL, system=system, messages=messages, max_tokens=64)

    assert _cache_config(first_client).system_instruction == SYSTEM
    assert isinstance(first, MessageResponse)
    assert first.usage.cache_creation_input_tokens == CACHED_TOKENS
    assert first.usage.cache_read_input_tokens == 0
    assert first.usage.input_tokens == PROMPT_TOKENS - CACHED_TOKENS
    assert isinstance(second, MessageResponse)
    assert second.usage.cache_creation_input_tokens is None
    assert second.usage.cache_read_input_tokens == CACHED_TOKENS
    assert second.usage.input_tokens == PROMPT_TOKENS - CACHED_TOKENS


@pytest.mark.asyncio
async def test_streamed_messages_bridge_reports_creation_then_read() -> None:
    usages = []
    for _ in range(2):
        client = _client()
        client.aio.models.generate_content_stream = AsyncMock(return_value=_aiter([_response()]))
        stream = await _provider(client).amessages(
            model=MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": "Q"}]}],
            system=SYSTEM,
            tools=[{"name": "get_weather", "input_schema": {"type": "object", "properties": {}}}],
            cache_control=EPHEMERAL,
            max_tokens=64,
            stream=True,
        )
        assert not isinstance(stream, MessageResponse)
        usages.append([event async for event in stream if event.type == "message_delta"][-1].usage)

    assert usages[0].cache_creation_input_tokens == CACHED_TOKENS
    assert usages[0].cache_read_input_tokens == 0
    assert usages[1].cache_creation_input_tokens is None
    assert usages[1].cache_read_input_tokens == CACHED_TOKENS


@pytest.mark.asyncio
async def test_messages_bridge_block_level_breakpoint_reaches_the_cache() -> None:
    client = _client()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Doc", "cache_control": EPHEMERAL}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Read it."}]},
        {"role": "user", "content": "Q"},
    ]

    await _provider(client).amessages(model=MODEL, messages=messages, max_tokens=64)

    cache_config = _cache_config(client)
    assert isinstance(cache_config.contents, list)
    assert _texts(cache_config.contents) == ["Doc"]


def test_find_last_breakpoint_ignores_non_ephemeral_marks() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "a", "cache_control": {"type": "persistent"}},
        {"role": "user", "content": "b", "extra_content": "not a dict"},
    ]

    assert find_last_breakpoint(messages) is None


def test_registry_evicts_the_least_recently_used_entry() -> None:
    lru = _CacheRegistry(max_entries=2)
    lru.put("a", _RegistryEntry("cachedContents/a", expires_at=100))
    lru.put("b", _RegistryEntry("cachedContents/b", expires_at=100))
    assert lru.get("a", now=0) is not None
    lru.put("c", _RegistryEntry("cachedContents/c", expires_at=100))

    assert lru.get("b", now=0) is None
    assert lru.get("a", now=0) is not None
    assert lru.get("c", now=0) is not None
    assert lru.get("a", now=100) is None


def test_client_identity_hashes_the_api_key_and_keeps_project_and_base_url() -> None:
    client = MagicMock()
    client._api_client.api_key = "secret-key"
    client._api_client.project = "my-project"
    client._api_client.location = "us-central1"
    client._api_client._http_options.base_url = "https://proxy.example"

    identity = client_identity(client)

    assert "secret-key" not in str(identity)
    assert identity["api_key_sha256"]
    assert identity["project"] == "my-project"
    assert identity["location"] == "us-central1"
    assert identity["base_url"] == "https://proxy.example"


def test_client_identity_without_an_api_key_or_http_options() -> None:
    client = MagicMock()
    client._api_client.api_key = None
    client._api_client._http_options = None

    identity = client_identity(client)

    assert identity["api_key_sha256"] is None
    assert identity["base_url"] is None


def test_report_cache_write_caps_the_write_at_prompt_tokens_and_uses_the_1h_bucket() -> None:
    usage = CompletionUsage(
        prompt_tokens=100,
        completion_tokens=1,
        total_tokens=101,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=100),
    )
    cache_use = ContextCacheUse(request_kwargs={}, uncached_kwargs={}, cache_write_tokens=150, ttl_seconds=3600)

    report_cache_write(usage, cache_use)

    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cache_write_tokens == 100
    assert usage.prompt_tokens_details.cached_tokens == 0
    assert usage.prompt_tokens_details.cache_creation_token_details is not None
    assert usage.prompt_tokens_details.cache_creation_token_details.ephemeral_1h_input_tokens == 100


def test_report_cache_write_without_a_write_or_usage_changes_nothing() -> None:
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11)

    report_cache_write(usage, ContextCacheUse(request_kwargs={}, uncached_kwargs={}))
    report_cache_write(None, ContextCacheUse(request_kwargs={}, uncached_kwargs={}, cache_write_tokens=5))

    assert usage.prompt_tokens_details is None
