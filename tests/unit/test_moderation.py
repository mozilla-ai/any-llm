from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai import OpenAIError

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
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "http://x"}},
    ]
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

    # Instantiate the provider separately so we can cleanly distinguish
    # "provider refused to construct due to config" (skip) from the
    # NotImplementedError path we actually want to assert.
    try:
        llm = AnyLLM.create(provider, api_key="test_key")
    except ImportError:
        pytest.skip(f"{provider.value} optional dependency missing, skipping")
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} requires additional config to instantiate, skipping")
    except OpenAIError as exc:
        pytest.skip(f"{provider.value} client init failed: {exc}")
    except (ValueError, TypeError) as exc:
        pytest.skip(f"{provider.value} requires additional config to instantiate: {exc}")
    except Exception as exc:
        # Catch provider-specific config errors raised during client
        # instantiation (e.g. botocore NoRegionError for bedrock).
        if type(exc).__name__ in {
            "NoRegionError",
            "NoCredentialsError",
            "ProfileNotFound",
            "DefaultCredentialsError",
        }:
            pytest.skip(f"{provider.value} requires additional config to instantiate: {exc}")
        raise

    with pytest.raises(NotImplementedError, match="does not support moderation"):
        await llm._amoderation("does-not-matter", "hello")


@pytest.mark.asyncio
async def test_mistral_rejects_multimodal_input() -> None:
    """Mistral must raise NotImplementedError with the locked multimodal phrase."""
    mistral_cls = AnyLLM.get_provider_class(LLMProvider.MISTRAL)
    inst = mistral_cls.__new__(mistral_cls)
    inst.client = Mock()  # type: ignore[attr-defined]

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
    inst.client = SimpleNamespace(moderations=moderations_api)  # type: ignore[assignment]

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
    inst.client = SimpleNamespace(moderations=moderations_api)  # type: ignore[assignment]

    response = await inst._amoderation("omni-moderation-latest", "hi", include_raw=True)
    assert response.results[0].provider_raw == {"full": "payload"}
    # include_raw should not be forwarded to the SDK call.
    moderations_api.create.assert_awaited_once_with(model="omni-moderation-latest", input="hi")


@pytest.mark.asyncio
async def test_openai_provider_amoderation_raises_when_unsupported() -> None:
    """Subclasses that opt out raise the locked NotImplementedError phrase."""
    from any_llm.providers.openai.openai import OpenaiProvider

    inst = OpenaiProvider.__new__(OpenaiProvider)
    inst.SUPPORTS_MODERATION = False
    inst.client = Mock()

    with pytest.raises(NotImplementedError, match="does not support moderation"):
        await inst._amoderation("foo", "bar")


def test_provider_metadata_includes_moderation_flag() -> None:
    """ProviderMetadata exposes the moderation capability."""
    metadata = AnyLLM.get_provider_class(LLMProvider.OPENAI).get_provider_metadata()
    assert metadata.moderation is True

    metadata = AnyLLM.get_provider_class(LLMProvider.OPENROUTER).get_provider_metadata()
    assert metadata.moderation is False


def test_moderation_sync_forwards_api_base_and_client_args() -> None:
    """Sync moderation forwards api_base and client_args to AnyLLM.create."""
    mock_response = _sample_response()
    mock_provider = Mock()
    mock_provider._moderation = Mock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        result = moderation(
            "omni-moderation-latest",
            input="hi",
            provider="openai",
            api_key="k",
            api_base="https://x",
            client_args={"timeout": 5},
        )

    call_args = mock_create.call_args
    assert call_args[0][0] == LLMProvider.OPENAI
    assert call_args[1]["api_base"] == "https://x"
    assert call_args[1]["timeout"] == 5
    assert result is mock_response


@pytest.mark.asyncio
async def test_amoderation_client_args_forwarded() -> None:
    """Async moderation passes client_args dict through to AnyLLM.create."""
    mock_provider = Mock()
    mock_provider._amoderation = AsyncMock(return_value=_sample_response())

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        await amoderation(
            "openai:omni-moderation-latest",
            input="hi",
            client_args={"timeout": 3},
        )

    assert mock_create.call_args[1]["timeout"] == 3


@pytest.mark.asyncio
async def test_amoderation_public_wrapper_delegates_to_internal() -> None:
    """AnyLLM.amoderation calls self._amoderation and propagates the result."""
    from any_llm.providers.openai.openai import OpenaiProvider

    inst = OpenaiProvider.__new__(OpenaiProvider)
    inst.client = Mock()

    sentinel = _sample_response()
    with patch.object(OpenaiProvider, "_amoderation", new=AsyncMock(return_value=sentinel)) as mock_inner:
        result = await inst.amoderation("omni-moderation-latest", "hi", include_raw=True)

    mock_inner.assert_awaited_once_with("omni-moderation-latest", "hi", include_raw=True)
    assert result is sentinel


@pytest.mark.asyncio
async def test_amoderation_base_raises_when_supported_but_unimplemented() -> None:
    """Subclass advertising SUPPORTS_MODERATION but not overriding hits the fallback."""
    # Call the base AnyLLM._amoderation directly, bypassing the subclass override.
    from any_llm.providers.openai.openai import OpenaiProvider

    inst = OpenaiProvider.__new__(OpenaiProvider)
    inst.client = Mock()

    with pytest.raises(NotImplementedError, match="Subclasses must implement _amoderation"):
        await AnyLLM._amoderation(inst, "m", "hi")


@pytest.mark.asyncio
async def test_mistral_amoderation_success_with_string_input() -> None:
    """Mistral provider wraps a string input into a list and calls moderate_async."""
    mistral_cls = AnyLLM.get_provider_class(LLMProvider.MISTRAL)
    inst = mistral_cls.__new__(mistral_cls)

    raw = SimpleNamespace(
        id=None,
        model=None,
        results=[
            SimpleNamespace(
                categories=SimpleNamespace(model_dump=lambda: {"violence": True, "hate": False}),
                category_scores=SimpleNamespace(model_dump=lambda: {"violence": 0.9, "hate": 0.02}),
                model_dump=lambda: {"raw": "payload"},
            )
        ],
    )

    classifiers = Mock()
    classifiers.moderate_async = AsyncMock(return_value=raw)
    inst.client = SimpleNamespace(classifiers=classifiers)  # type: ignore[attr-defined]

    # Empty model name exercises the default fallback branch.
    response = await inst._amoderation("", "hurt me")

    classifiers.moderate_async.assert_awaited_once()
    call_kwargs = classifiers.moderate_async.call_args.kwargs
    assert call_kwargs["model"] == "mistral-moderation-latest"
    assert call_kwargs["inputs"] == ["hurt me"]
    assert response.results[0].flagged is True
    assert response.results[0].provider_raw is None
    assert response.id.startswith("modr-")


@pytest.mark.asyncio
async def test_mistral_amoderation_list_input_and_include_raw() -> None:
    """List-of-strings input is forwarded as-is; include_raw populates provider_raw."""
    mistral_cls = AnyLLM.get_provider_class(LLMProvider.MISTRAL)
    inst = mistral_cls.__new__(mistral_cls)

    raw = SimpleNamespace(
        id="mod-xyz",
        model="mistral-moderation-latest",
        results=[
            SimpleNamespace(
                categories=SimpleNamespace(model_dump=lambda: {"violence": False}),
                category_scores=SimpleNamespace(model_dump=lambda: {"violence": 0.01}),
                model_dump=lambda: {"raw": "payload"},
            )
        ],
    )

    classifiers = Mock()
    classifiers.moderate_async = AsyncMock(return_value=raw)
    inst.client = SimpleNamespace(classifiers=classifiers)  # type: ignore[attr-defined]

    response = await inst._amoderation(
        "mistral-moderation-latest",
        ["a", "b"],
        include_raw=True,
    )

    call_kwargs = classifiers.moderate_async.call_args.kwargs
    assert call_kwargs["inputs"] == ["a", "b"]
    assert "include_raw" not in call_kwargs
    assert response.id == "mod-xyz"
    assert response.results[0].provider_raw == {"raw": "payload"}


def test_mistral_as_plain_dict_handles_all_branches() -> None:
    """_as_plain_dict covers None, mapping, coercible, and fallback paths."""
    from any_llm.providers.mistral.utils import _as_plain_dict

    assert _as_plain_dict(None) == {}
    assert _as_plain_dict({"a": 1}) == {"a": 1}
    # Iterable of key/value pairs is coerceable by dict().
    assert _as_plain_dict([("x", 1), ("y", 2)]) == {"x": 1, "y": 2}
    # A pydantic-like object whose model_dump returns a non-dict drops to {}.
    bad = SimpleNamespace(model_dump=lambda: "not a dict")
    assert _as_plain_dict(bad) == {}
    # Object that cannot be dict-coerced falls through to {}.
    assert _as_plain_dict(123) == {}


def test_openai_moderation_converter_handles_types_non_dict_and_plain_dict_inputs() -> None:
    """Covers _dump_if_model dict passthrough and applied_types non-dict branch."""
    from any_llm.providers.openai.utils import _convert_moderation_response_from_openai

    raw = SimpleNamespace(
        id="modr-2",
        model="omni-moderation-latest",
        results=[
            SimpleNamespace(
                flagged=False,
                # plain dicts exercise the no-model_dump path of _dump_if_model
                categories={"violence": False, "hate": True},
                category_scores={"violence": 0.01, "hate": 0.9},
                category_applied_input_types="not-a-dict",
                model_dump=lambda: {"raw": "payload"},
            )
        ],
    )

    result = _convert_moderation_response_from_openai(raw, include_raw=False)
    assert result.results[0].categories == {"violence": False, "hate": True}
    assert result.results[0].category_scores == {"violence": 0.01, "hate": 0.9}
    # Non-dict types_dump collapses to None.
    assert result.results[0].category_applied_input_types is None


@pytest.mark.asyncio
async def test_gateway_platform_amoderation_wraps_errors() -> None:
    """Gateway platform-mode _amoderation wraps provider errors."""
    from any_llm.exceptions import AuthenticationError
    from any_llm.providers.gateway.gateway import GatewayProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = GatewayProvider(
            api_key="token",
            api_base="https://gateway.example.com",
            platform_mode=True,
        )

    request = Mock()
    response = Mock(status_code=401, headers={})
    import openai

    exc = openai.APIStatusError("Unauthorized", response=response, body={"message": "Unauthorized"})
    exc.request = request

    with patch.object(type(provider).__bases__[0], "_amoderation", new_callable=AsyncMock, side_effect=exc):
        with pytest.raises(AuthenticationError):
            await provider._amoderation("omni-moderation-latest", "hi")


@pytest.mark.asyncio
async def test_gateway_platform_amoderation_success() -> None:
    """Gateway platform-mode _amoderation returns the result on success."""
    from any_llm.providers.gateway.gateway import GatewayProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = GatewayProvider(
            api_key="token",
            api_base="https://gateway.example.com",
            platform_mode=True,
        )

    sentinel = _sample_response()
    with patch.object(
        type(provider).__bases__[0], "_amoderation", new_callable=AsyncMock, return_value=sentinel
    ) as mock_super:
        result = await provider._amoderation("omni-moderation-latest", "hi")

    mock_super.assert_awaited_once()
    assert result is sentinel


@pytest.mark.asyncio
async def test_gateway_non_platform_amoderation_no_wrapping() -> None:
    """In non-platform mode, _amoderation errors pass through unchanged."""
    import openai

    from any_llm.providers.gateway.gateway import GatewayProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = GatewayProvider(api_key="key", api_base="https://gateway.example.com")

    response = Mock(status_code=502, headers={})
    exc = openai.APIStatusError("Bad gateway", response=response, body={})
    exc.request = Mock()

    with patch.object(type(provider).__bases__[0], "_amoderation", new_callable=AsyncMock, side_effect=exc):
        with pytest.raises(openai.APIStatusError):
            await provider._amoderation("omni-moderation-latest", "hi")
