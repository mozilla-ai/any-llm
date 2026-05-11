from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm import AnyLLM
from any_llm.api import aimage_generation, image_generation
from any_llm.constants import LLMProvider
from any_llm.types.image import ImageGenerationParams, ImagesResponse


def _make_mock_images_response() -> ImagesResponse:
    return ImagesResponse(created=1234567890, data=[])


@pytest.mark.asyncio
async def test_aimage_generation_with_api_config() -> None:
    mock_provider = Mock()
    mock_response = _make_mock_images_response()
    mock_provider._aimage_generation = AsyncMock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = await aimage_generation(
            "openai:dall-e-3",
            prompt="A cat",
            api_key="test_key",
            api_base="https://test.example.com",
        )

        call_args = mock_create.call_args
        assert call_args[0][0] == LLMProvider.OPENAI
        assert call_args[1]["api_key"] == "test_key"
        assert call_args[1]["api_base"] == "https://test.example.com"

        mock_provider._aimage_generation.assert_called_once()
        params = mock_provider._aimage_generation.call_args[0][0]
        assert isinstance(params, ImageGenerationParams)
        assert params.model_id == "dall-e-3"
        assert params.prompt == "A cat"
        assert result == mock_response


@pytest.mark.asyncio
async def test_aimage_generation_with_explicit_provider() -> None:
    mock_provider = Mock()
    mock_response = _make_mock_images_response()
    mock_provider._aimage_generation = AsyncMock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = await aimage_generation(
            "dall-e-3",
            prompt="A dog",
            provider="openai",
            size="1024x1024",
            quality="hd",
        )

        call_args = mock_create.call_args
        assert call_args[0][0] == LLMProvider.OPENAI

        params = mock_provider._aimage_generation.call_args[0][0]
        assert params.model_id == "dall-e-3"
        assert params.prompt == "A dog"
        assert params.size == "1024x1024"
        assert params.quality == "hd"
        assert result == mock_response


@pytest.mark.asyncio
async def test_aimage_generation_passes_all_params() -> None:
    mock_provider = Mock()
    mock_response = _make_mock_images_response()
    mock_provider._aimage_generation = AsyncMock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        await aimage_generation(
            "openai:dall-e-3",
            prompt="A bird",
            n=2,
            size="1792x1024",
            quality="standard",
            style="natural",
            response_format="b64_json",
            user="user-123",
        )

        params = mock_provider._aimage_generation.call_args[0][0]
        assert params.n == 2
        assert params.size == "1792x1024"
        assert params.quality == "standard"
        assert params.style == "natural"
        assert params.response_format == "b64_json"
        assert params.user == "user-123"


def test_sync_image_generation_dispatches() -> None:
    mock_provider = Mock()
    mock_response = _make_mock_images_response()
    mock_provider._image_generation = Mock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = image_generation(
            "openai:dall-e-3",
            prompt="A fish",
            api_key="test_key",
        )

        mock_provider._image_generation.assert_called_once()
        assert result == mock_response


@pytest.mark.asyncio
async def test_aimage_generation_unsupported_provider_raises_not_implemented() -> None:
    params = ImageGenerationParams(model_id="some-model", prompt="A cat")
    base = Mock(spec=AnyLLM)
    base.SUPPORTS_IMAGE_GENERATION = False
    with pytest.raises(NotImplementedError, match="doesn't support image generation"):
        await AnyLLM._aimage_generation(base, params)


def test_image_generation_params_to_api_kwargs_excludes_none() -> None:
    params = ImageGenerationParams(model_id="dall-e-3", prompt="A cat")
    kwargs = params.to_api_kwargs()
    assert "model_id" not in kwargs
    assert kwargs == {"prompt": "A cat"}


def test_image_generation_params_to_api_kwargs_includes_set_values() -> None:
    params = ImageGenerationParams(
        model_id="dall-e-3",
        prompt="A cat",
        n=2,
        size="1024x1024",
        quality="hd",
        style="vivid",
        response_format="url",
        user="u1",
    )
    kwargs = params.to_api_kwargs()
    assert "model_id" not in kwargs
    assert kwargs == {
        "prompt": "A cat",
        "n": 2,
        "size": "1024x1024",
        "quality": "hd",
        "style": "vivid",
        "response_format": "url",
        "user": "u1",
    }


def test_image_generation_params_rejects_extra_fields() -> None:
    with pytest.raises(Exception, match="extra"):
        ImageGenerationParams(model_id="dall-e-3", prompt="A cat", bogus="value")  # type: ignore[call-arg]


def test_supports_image_generation_only_on_expected_providers() -> None:
    expected_supported = {LLMProvider.OPENAI, LLMProvider.AZUREOPENAI, LLMProvider.GATEWAY}

    for provider_enum in expected_supported:
        cls = AnyLLM.get_provider_class(provider_enum)
        assert cls.SUPPORTS_IMAGE_GENERATION is True, f"{provider_enum.value} should support image generation"

    for provider_enum in LLMProvider:
        if provider_enum in expected_supported:
            continue
        try:
            cls = AnyLLM.get_provider_class(provider_enum)
        except ImportError:
            continue
        assert cls.SUPPORTS_IMAGE_GENERATION is False, f"{provider_enum.value} should not support image generation"
