from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm import AnyLLM, amoderation, moderation
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.moderation import ModerationResponse, ModerationResult


def _sample_response() -> ModerationResponse:
    return ModerationResponse(
        id="modr-test",
        model="omni-moderation-latest",
        results=[
            ModerationResult(
                flagged=True,
                categories={"violence": True, "hate": False},
                category_scores={"violence": 0.93, "hate": 0.01},
                category_applied_input_types={"violence": ["text"]},
            )
        ],
    )


@pytest.mark.asyncio
async def test_amoderation_with_api_config() -> None:
    """amoderation forwards api_key / api_base to AnyLLM.create and returns the provider response."""
    mock_response = _sample_response()
    mock_provider = Mock()
    mock_provider._amoderation = AsyncMock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = await amoderation(
            "openai:omni-moderation-latest",
            input="I want to hurt someone",
            api_key="test_key",
            api_base="https://test.example.com",
        )

    call_args = mock_create.call_args
    assert call_args[0][0] == LLMProvider.OPENAI
    assert call_args[1]["api_key"] == "test_key"
    assert call_args[1]["api_base"] == "https://test.example.com"
    mock_provider._amoderation.assert_awaited_once_with("omni-moderation-latest", "I want to hurt someone")
    assert result is mock_response


@pytest.mark.asyncio
async def test_amoderation_explicit_provider_kwarg() -> None:
    """Passing provider= separately skips provider:model parsing."""
    mock_provider = Mock()
    mock_provider._amoderation = AsyncMock(return_value=_sample_response())

    with patch("any_llm.any_llm.AnyLLM.create", return_value=mock_provider):
        await amoderation(
            "omni-moderation-latest",
            input="be kind",
            provider="openai",
            api_key="k",
        )

    mock_provider._amoderation.assert_awaited_once_with("omni-moderation-latest", "be kind")


@pytest.mark.asyncio
async def test_amoderation_multi_input() -> None:
    """List-of-strings input flows through without coercion."""
    mock_provider = Mock()
    mock_provider._amoderation = AsyncMock(return_value=_sample_response())

    with patch("any_llm.any_llm.AnyLLM.create", return_value=mock_provider):
        await amoderation(
            "openai:omni-moderation-latest",
            input=["first", "second"],
            api_key="k",
        )

    mock_provider._amoderation.assert_awaited_once_with("omni-moderation-latest", ["first", "second"])


@pytest.mark.asyncio
async def test_amoderation_multimodal_input_passthrough() -> None:
    """List-of-dict (multimodal) input is forwarded unchanged."""
    parts = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "http://x"}}]
    mock_provider = Mock()
    mock_provider._amoderation = AsyncMock(return_value=_sample_response())

    with patch("any_llm.any_llm.AnyLLM.create", return_value=mock_provider):
        await amoderation("openai:omni-moderation-latest", input=parts, api_key="k")

    mock_provider._amoderation.assert_awaited_once_with("omni-moderation-latest", parts)


@pytest.mark.asyncio
async def test_amoderation_forwards_include_raw_kwarg() -> None:
    """Extra kwargs (include_raw) reach the provider's _amoderation call."""
    mock_provider = Mock()
    mock_provider._amoderation = AsyncMock(return_value=_sample_response())

    with patch("any_llm.any_llm.AnyLLM.create", return_value=mock_provider):
        await amoderation(
            "openai:omni-moderation-latest",
            input="hi",
            api_key="k",
            include_raw=True,
        )

    mock_provider._amoderation.assert_awaited_once_with("omni-moderation-latest", "hi", include_raw=True)


def test_moderation_sync_wraps_amoderation() -> None:
    """Sync helper wraps the async helper via run_async_in_sync."""
    mock_response = _sample_response()
    mock_provider = Mock()
    mock_provider._moderation = Mock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create", return_value=mock_provider):
        result = moderation(
            "openai:omni-moderation-latest",
            input="be kind",
            api_key="k",
        )

    mock_provider._moderation.assert_called_once_with("omni-moderation-latest", "be kind")
    assert result is mock_response


@pytest.mark.asyncio
async def test_amoderation_unsupported_provider_raises_not_implemented(provider: LLMProvider) -> None:
    """Providers that opt out must raise NotImplementedError with the locked phrase."""
    try:
        cls = AnyLLM.get_provider_class(provider)
    except ImportError:
        pytest.skip(f"{provider.value} optional dependency missing, skipping")
    if cls.SUPPORTS_MODERATION:
        pytest.skip(f"{provider.value} supports moderation, skipping")

    try:
        with pytest.raises(NotImplementedError, match="does not support moderation"):
            await amoderation(
                f"{provider.value}:does-not-matter",
                input="hello",
                api_key="test_key",
            )
    except ImportError:
        pytest.skip(f"{provider.value} optional dependency missing, skipping")
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} requires additional config to instantiate, skipping")


@pytest.mark.asyncio
async def test_mistral_rejects_multimodal_input() -> None:
    """Mistral must raise NotImplementedError with the locked multimodal phrase."""
    mistral_cls = AnyLLM.get_provider_class(LLMProvider.MISTRAL)
    inst = mistral_cls.__new__(mistral_cls)
    inst.client = Mock()  # unused on the raise path

    with pytest.raises(NotImplementedError, match="does not support multimodal moderation input"):
        await inst._amoderation(
            "mistral-moderation-latest",
            [{"type": "text", "text": "hi"}],
        )


def test_openai_moderation_converter_filters_missing_keys() -> None:
    """Absent (None) keys are dropped from categories / category_scores."""
    from any_llm.providers.openai.utils import _convert_moderation_response_from_openai

    raw = SimpleNamespace(
        id="modr-123",
        model="omni-moderation-latest",
        results=[
            SimpleNamespace(
                flagged=True,
                categories=SimpleNamespace(model_dump=lambda: {"violence": True, "hate": None, "self_harm": False}),
                category_scores=SimpleNamespace(model_dump=lambda: {"violence": 0.9, "hate": None, "self_harm": 0.02}),
                category_applied_input_types=SimpleNamespace(model_dump=lambda: {"violence": ["text"], "hate": None}),
                model_dump=lambda: {"raw": "payload"},
            )
        ],
    )

    result = _convert_moderation_response_from_openai(raw, include_raw=True)

    assert result.id == "modr-123"
    assert result.results[0].flagged is True
    assert result.results[0].categories == {"violence": True, "self_harm": False}
    assert result.results[0].category_scores == {"violence": 0.9, "self_harm": 0.02}
    assert result.results[0].category_applied_input_types == {"violence": ["text"]}
    assert result.results[0].provider_raw == {"raw": "payload"}


def test_openai_moderation_converter_include_raw_default_false() -> None:
    """provider_raw stays None unless include_raw=True."""
    from any_llm.providers.openai.utils import _convert_moderation_response_from_openai

    raw = SimpleNamespace(
        id="modr-1",
        model="omni-moderation-latest",
        results=[
            SimpleNamespace(
                flagged=False,
                categories=SimpleNamespace(model_dump=lambda: {"violence": False}),
                category_scores=SimpleNamespace(model_dump=lambda: {"violence": 0.01}),
                category_applied_input_types=None,
                model_dump=lambda: {"raw": "payload"},
            )
        ],
    )

    result = _convert_moderation_response_from_openai(raw, include_raw=False)
    assert result.results[0].provider_raw is None
    assert result.results[0].category_applied_input_types is None


def test_mistral_moderation_converter_synthesizes_flagged() -> None:
    """Mistral has no flagged field; helper derives it from categories."""
    from any_llm.providers.mistral.utils import _create_openai_moderation_response_from_mistral

    raw = SimpleNamespace(
        id="mod-abc",
        model="mistral-moderation-latest",
        results=[
            SimpleNamespace(
                categories=SimpleNamespace(model_dump=lambda: {"violence": True, "hate": False}),
                category_scores=SimpleNamespace(model_dump=lambda: {"violence": 0.8, "hate": 0.05}),
                model_dump=lambda: {"raw": "payload"},
            ),
            SimpleNamespace(
                categories=SimpleNamespace(model_dump=lambda: {"violence": False, "hate": False}),
                category_scores=SimpleNamespace(model_dump=lambda: {"violence": 0.02, "hate": 0.01}),
                model_dump=lambda: {"raw": "clean"},
            ),
        ],
    )

    result = _create_openai_moderation_response_from_mistral(raw, include_raw=True)
    assert result.id == "mod-abc"
    assert result.model == "mistral-moderation-latest"
    assert result.results[0].flagged is True
    assert result.results[1].flagged is False
    assert result.results[0].provider_raw == {"raw": "payload"}
    assert result.results[0].category_applied_input_types is None


def test_mistral_moderation_converter_respects_include_raw_false() -> None:
    from any_llm.providers.mistral.utils import _create_openai_moderation_response_from_mistral

    raw = SimpleNamespace(
        id="mod-abc",
        model=None,
        results=[
            SimpleNamespace(
                categories=SimpleNamespace(model_dump=lambda: {"violence": False}),
                category_scores=SimpleNamespace(model_dump=lambda: {"violence": 0.1}),
                model_dump=lambda: {"raw": "payload"},
            ),
        ],
    )

    result = _create_openai_moderation_response_from_mistral(raw, include_raw=False)
    # model falls back to the default label when the SDK returns None
    assert result.model == "mistral-moderation-latest"
    assert result.results[0].provider_raw is None


@pytest.mark.asyncio
async def test_openai_provider_amoderation_defaults_and_converts() -> None:
    """BaseOpenAIProvider forwards to client.moderations.create and converts the response."""
    from any_llm.providers.openai.openai import OpenaiProvider

    inst = OpenaiProvider.__new__(OpenaiProvider)
    moderations_api = Mock()
    raw_response = SimpleNamespace(
        id="modr-xyz",
        model="omni-moderation-latest",
        results=[
            SimpleNamespace(
                flagged=True,
                categories=SimpleNamespace(model_dump=lambda: {"violence": True}),
                category_scores=SimpleNamespace(model_dump=lambda: {"violence": 0.99}),
                category_applied_input_types=None,
                model_dump=lambda: {"raw": True},
            )
        ],
    )
    moderations_api.create = AsyncMock(return_value=raw_response)
    inst.client = SimpleNamespace(moderations=moderations_api)

    # Empty model should default to omni-moderation-latest.
    response = await inst._amoderation("", "hurt me")
    moderations_api.create.assert_awaited_once_with(model="omni-moderation-latest", input="hurt me")
    assert response.id == "modr-xyz"
    assert response.results[0].flagged is True
    assert response.results[0].provider_raw is None


@pytest.mark.asyncio
async def test_openai_provider_amoderation_include_raw() -> None:
    """include_raw=True populates provider_raw on each result."""
    from any_llm.providers.openai.openai import OpenaiProvider

    inst = OpenaiProvider.__new__(OpenaiProvider)
    moderations_api = Mock()
    moderations_api.create = AsyncMock(
        return_value=SimpleNamespace(
            id="modr-1",
            model="omni-moderation-latest",
            results=[
                SimpleNamespace(
                    flagged=False,
                    categories=SimpleNamespace(model_dump=lambda: {"violence": False}),
                    category_scores=SimpleNamespace(model_dump=lambda: {"violence": 0.01}),
                    category_applied_input_types=None,
                    model_dump=lambda: {"full": "payload"},
                )
            ],
        )
    )
    inst.client = SimpleNamespace(moderations=moderations_api)

    response = await inst._amoderation("omni-moderation-latest", "hi", include_raw=True)
    assert response.results[0].provider_raw == {"full": "payload"}
    # include_raw should not be forwarded to the SDK call.
    moderations_api.create.assert_awaited_once_with(model="omni-moderation-latest", input="hi")


@pytest.mark.asyncio
async def test_openai_provider_amoderation_raises_when_unsupported() -> None:
    """Subclasses that opt out raise the locked NotImplementedError phrase."""
    from any_llm.providers.openai.openai import OpenaiProvider

    inst = OpenaiProvider.__new__(OpenaiProvider)
    inst.SUPPORTS_MODERATION = False  # type: ignore[misc]
    inst.client = Mock()

    with pytest.raises(NotImplementedError, match="does not support moderation"):
        await inst._amoderation("foo", "bar")


def test_provider_metadata_includes_moderation_flag() -> None:
    """ProviderMetadata exposes the moderation capability."""
    metadata = AnyLLM.get_provider_class(LLMProvider.OPENAI).get_provider_metadata()
    assert metadata.moderation is True

    metadata = AnyLLM.get_provider_class(LLMProvider.OPENROUTER).get_provider_metadata()
    assert metadata.moderation is False
