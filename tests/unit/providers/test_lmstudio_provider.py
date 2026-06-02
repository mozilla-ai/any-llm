from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm import AnyLLM
from any_llm.providers.lmstudio import utils
from any_llm.providers.lmstudio.lmstudio import LmstudioProvider
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
    CreateEmbeddingResponse,
)


def _make_stats(
    *,
    prompt: int = 10,
    predicted: int = 5,
    total: int = 15,
    stop_reason: str = "eosFound",
) -> Mock:
    stats = Mock()
    stats.prompt_tokens_count = prompt
    stats.predicted_tokens_count = predicted
    stats.total_tokens_count = total
    stats.stop_reason = stop_reason
    return stats


def _make_prediction_result(content: str = "Hello there", model_key: str = "qwen3-0.6b") -> Mock:
    result = Mock()
    result.content = content
    result.stats = _make_stats()
    result.model_info = Mock()
    result.model_info.model_key = model_key
    return result


class _FakeStream:
    """Minimal async-iterable stand-in for lmstudio's AsyncPredictionStream."""

    def __init__(self, fragments: list[Mock], result: Mock) -> None:
        self._fragments = fragments
        self._result = result

    async def __aiter__(self) -> AsyncIterator[Mock]:
        for fragment in self._fragments:
            yield fragment

    def result(self) -> Mock:
        return self._result


def _patch_async_client(inner_client: Mock) -> Any:
    """Patch AsyncClient so `async with AsyncClient(...)` yields ``inner_client``."""
    client_cm = Mock()
    client_cm.__aenter__ = AsyncMock(return_value=inner_client)
    client_cm.__aexit__ = AsyncMock(return_value=None)
    return patch(
        "any_llm.providers.lmstudio.lmstudio.AsyncClient",
        return_value=client_cm,
    )


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    p = LmstudioProvider()
    assert p.PROVIDER_NAME == "lmstudio"
    assert p.api_host == "localhost:1234"
    assert p.SUPPORTS_COMPLETION is True
    assert p.SUPPORTS_COMPLETION_STREAMING is True
    assert p.SUPPORTS_COMPLETION_REASONING is True
    assert p.SUPPORTS_EMBEDDING is True
    assert p.SUPPORTS_LIST_MODELS is True
    assert p.SUPPORTS_RESPONSES is False


def test_api_key_not_required() -> None:
    """Test that LM Studio does not require an API key."""
    p = LmstudioProvider()
    assert p.PROVIDER_NAME == "lmstudio"


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        ("http://localhost:1234/v1", "localhost:1234"),
        ("https://example.com:4321/v1", "example.com:4321"),
        ("ws://10.0.0.2:9999", "10.0.0.2:9999"),
        ("192.168.0.5:5000", "192.168.0.5:5000"),
        ("http://localhost:1234/", "localhost:1234"),
        (None, None),
        ("", None),
    ],
)
def test_normalize_api_host(api_base: str | None, expected: str | None) -> None:
    assert LmstudioProvider._normalize_api_host(api_base) == expected


def test_api_base_override_sets_host() -> None:
    p = LmstudioProvider(api_base="http://10.0.0.2:9999/v1")
    assert p.api_host == "10.0.0.2:9999"


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    p = AnyLLM.create("lmstudio")
    assert isinstance(p, LmstudioProvider)
    assert p.PROVIDER_NAME == "lmstudio"

    supported = AnyLLM.get_supported_providers()
    assert "lmstudio" in supported


def test_model_provider_split() -> None:
    """Test that model string parsing works correctly."""
    provider_enum, model_name = AnyLLM.split_model_provider("lmstudio:google/gemma-3-4b")
    assert provider_enum.value == "lmstudio"
    assert model_name == "google/gemma-3-4b"


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = LmstudioProvider.get_provider_metadata()
    assert metadata.name == "lmstudio"
    assert metadata.env_key == "LM_STUDIO_API_KEY"
    assert metadata.doc_url == "https://lmstudio.ai/docs/python"
    assert metadata.completion is True
    assert metadata.streaming is True
    assert metadata.responses is False


def test_build_chat_maps_roles() -> None:
    p = LmstudioProvider()
    chat = p._build_chat(
        [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "ignored"}},
                ],
            },
        ]
    )
    # 1 system + 3 conversation turns
    assert len(chat._messages) == 4


def test_extract_text_content_variants() -> None:
    assert LmstudioProvider._extract_text_content(None) == ""
    assert LmstudioProvider._extract_text_content("plain") == "plain"
    assert (
        LmstudioProvider._extract_text_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a b"
    )


def test_convert_completion_params_maps_known_fields() -> None:
    params = CompletionParams(
        model_id="qwen3-0.6b",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=50,
        top_p=0.9,
        stop="STOP",
    )
    config = LmstudioProvider._convert_completion_params(params)
    assert config["temperature"] == 0.5
    assert config["maxTokens"] == 50
    assert config["topPSampling"] == 0.9
    assert config["stopStrings"] == ["STOP"]


def test_convert_completion_params_stop_list_and_extra_kwargs() -> None:
    params = CompletionParams(
        model_id="qwen3-0.6b",
        messages=[{"role": "user", "content": "hi"}],
        stop=["a", "b"],
    )
    config = LmstudioProvider._convert_completion_params(params, topKSampling=40)
    assert config["stopStrings"] == ["a", "b"]
    assert config["topKSampling"] == 40


def test_resolve_response_format_dict_passthrough() -> None:
    params = CompletionParams(
        model_id="qwen3-0.6b",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    assert LmstudioProvider._resolve_response_format(params) == {"type": "json_object"}


def test_resolve_response_format_pydantic() -> None:
    from pydantic import BaseModel

    class Out(BaseModel):
        name: str

    params = CompletionParams(
        model_id="qwen3-0.6b",
        messages=[{"role": "user", "content": "hi"}],
        response_format=Out,
    )
    schema = LmstudioProvider._resolve_response_format(params)
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "name" in schema["properties"]


def test_resolve_response_format_none() -> None:
    params = CompletionParams(model_id="m", messages=[{"role": "user", "content": "hi"}])
    assert LmstudioProvider._resolve_response_format(params) is None


@pytest.mark.asyncio
async def test_acompletion_non_stream() -> None:
    provider = LmstudioProvider()

    model_handle = Mock()
    model_handle.respond = AsyncMock(return_value=_make_prediction_result())
    inner_client = Mock()
    inner_client.llm.model = AsyncMock(return_value=model_handle)

    with _patch_async_client(inner_client):
        result = await provider._acompletion(
            CompletionParams(
                model_id="qwen3-0.6b",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.2,
            )
        )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Hello there"
    inner_client.llm.model.assert_awaited_once_with("qwen3-0.6b")
    call_kwargs = model_handle.respond.call_args.kwargs
    assert call_kwargs["config"] == {"temperature": 0.2}


@pytest.mark.asyncio
async def test_acompletion_extracts_reasoning() -> None:
    provider = LmstudioProvider()

    model_handle = Mock()
    model_handle.respond = AsyncMock(return_value=_make_prediction_result(content="<think>because</think>final"))
    inner_client = Mock()
    inner_client.llm.model = AsyncMock(return_value=model_handle)

    with _patch_async_client(inner_client):
        result = await provider._acompletion(
            CompletionParams(model_id="m", messages=[{"role": "user", "content": "Hello"}])
        )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "final"
    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content == "because"


@pytest.mark.asyncio
async def test_acompletion_streaming() -> None:
    provider = LmstudioProvider()

    frag_reasoning = Mock(reasoning_type="reasoning", content="thinking")
    frag_content = Mock(reasoning_type="none", content="hello")
    stream = _FakeStream([frag_reasoning, frag_content], _make_prediction_result())

    model_handle = Mock()
    model_handle.respond_stream = AsyncMock(return_value=stream)
    inner_client = Mock()
    inner_client.llm.model = AsyncMock(return_value=model_handle)

    with _patch_async_client(inner_client):
        result = await provider._acompletion(
            CompletionParams(
                model_id="m",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
        )
        chunks = [chunk async for chunk in result]  # type: ignore[union-attr]

    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    # reasoning fragment + content fragment + final chunk
    assert len(chunks) == 3
    assert chunks[0].choices[0].delta.reasoning is not None
    assert chunks[1].choices[0].delta.content == "hello"
    assert chunks[2].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_acompletion_with_tools_raises() -> None:
    provider = LmstudioProvider()

    def get_weather(location: str) -> str:
        return "sunny"

    with pytest.raises(NotImplementedError, match="tool calling"):
        await provider._acompletion(
            CompletionParams(
                model_id="m",
                messages=[{"role": "user", "content": "weather?"}],
                tools=[get_weather],
            )
        )


@pytest.mark.asyncio
async def test_aembedding_single_string() -> None:
    provider = LmstudioProvider()

    embedding_handle = Mock()
    embedding_handle.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    inner_client = Mock()
    inner_client.embedding.model = AsyncMock(return_value=embedding_handle)

    with _patch_async_client(inner_client):
        response = await provider._aembedding("nomic-embed", "hello")

    assert isinstance(response, CreateEmbeddingResponse)
    assert len(response.data) == 1
    assert response.data[0].embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_aembedding_list_input() -> None:
    provider = LmstudioProvider()

    embedding_handle = Mock()
    embedding_handle.embed = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    inner_client = Mock()
    inner_client.embedding.model = AsyncMock(return_value=embedding_handle)

    with _patch_async_client(inner_client):
        response = await provider._aembedding("nomic-embed", ["a", "b"])

    assert len(response.data) == 2
    assert response.data[1].index == 1
    assert response.data[1].embedding == [0.3, 0.4]


@pytest.mark.asyncio
async def test_alist_models() -> None:
    provider = LmstudioProvider()

    inner_client = Mock()
    inner_client.list_downloaded_models = AsyncMock(
        return_value=[Mock(model_key="qwen3-0.6b"), Mock(model_key="nomic-embed")]
    )

    with _patch_async_client(inner_client):
        models = await provider._alist_models()

    assert [m.id for m in models] == ["qwen3-0.6b", "nomic-embed"]
    assert all(m.owned_by == "lmstudio" for m in models)


def test_convert_models_list_helper() -> None:
    models = utils.convert_models_list([Mock(model_key="a"), Mock(model_key="b")])
    assert [m.id for m in models] == ["a", "b"]


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("eosFound", "stop"),
        ("stopStringFound", "stop"),
        ("userStopped", "stop"),
        ("maxPredictedTokensReached", "length"),
        ("contextLengthReached", "length"),
        ("toolCalls", "tool_calls"),
        ("unknown-reason", "stop"),
        (None, "stop"),
    ],
)
def test_map_stop_reason(stop_reason: str | None, expected: str) -> None:
    assert utils._map_stop_reason(stop_reason) == expected


@pytest.mark.parametrize(
    ("content", "expected_content", "expected_reasoning"),
    [
        ("<think>because</think>final", "final", "because"),
        ("preface <think>because</think> final", "preface  final", "because"),
        ("no tags here", "no tags here", None),
    ],
)
def test_split_reasoning_from_content(content: str, expected_content: str, expected_reasoning: str | None) -> None:
    assert utils._split_reasoning_from_content(content) == (expected_content, expected_reasoning)
