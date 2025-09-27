import base64
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm import acompletion
from any_llm.provider import ClientConfig, Provider, ProviderFactory, ProviderName
from any_llm.types.completion import ChatCompletionMessage, CompletionParams, Reasoning


@pytest.mark.asyncio
async def test_completion_valid_base64_image() -> None:
    """Test completion accepts valid base64 image data."""
    # base64 already imported at top

    valid_png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    mock_provider = Mock()
    mock_provider.acompletion = AsyncMock()
    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["gemini"]
        mock_factory.get_provider_enum.return_value = ProviderName.OPENAI
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "gemini-pro")
        mock_factory.create_provider.return_value = mock_provider
        await acompletion(
            "gemini-pro", messages=[{"role": "user", "content": "Describe image", "image_base64": valid_png}]
        )
        mock_provider.acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_completion_invalid_base64_image() -> None:
    """Test completion raises error for invalid base64 image data."""
    invalid_base64 = "not_base64!"
    with pytest.raises(ValueError, match="Invalid base64 image data supplied."):
        await acompletion(
            "gemini-pro", messages=[{"role": "user", "content": "Describe image", "image_base64": invalid_base64}]
        )


@pytest.mark.asyncio
async def test_completion_invalid_image_url() -> None:
    """Test completion raises error for invalid image URL."""
    with pytest.raises(ValueError, match="Invalid image URL supplied."):
        await acompletion(
            "vertexai-pro", messages=[{"role": "user", "content": "Describe image", "image_url": "ftp://invalid-url"}]
        )


@pytest.mark.asyncio
async def test_completion_invalid_image_bytes_type() -> None:
    """Test completion raises error for non-bytes image_bytes."""
    with pytest.raises(ValueError, match="image_bytes must be bytes or bytearray."):
        await acompletion(
            "gemini-pro", messages=[{"role": "user", "content": "Describe image", "image_bytes": "not_bytes"}]
        )


@pytest.mark.asyncio
async def test_completion_invalid_model_format_no_slash() -> None:
    """Test completion raises ValueError for model without separator."""
    with pytest.raises(
        ValueError, match="Invalid model format. Expected 'provider:model' or 'provider/model', got 'gpt-4'"
    ):
        await acompletion("gpt-4", messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_completion_invalid_model_format_empty_provider() -> None:
    """Test completion raises ValueError for model with empty provider."""
    with pytest.raises(ValueError, match="Invalid model format"):
        await acompletion("/model", messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_completion_invalid_model_format_empty_model() -> None:
    """Test completion raises ValueError for model with empty model name."""
    with pytest.raises(ValueError, match="Invalid model format"):
        await acompletion("provider/", messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_completion_invalid_model_format_multiple_slashes() -> None:
    """Test completion handles multiple slashes correctly (should work - takes first split)."""
    mock_provider = Mock()
    mock_provider.acompletion = AsyncMock()

    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["provider"]
        mock_factory.get_provider_enum.return_value = ProviderName.OPENAI  # Using a valid provider
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "model/extra")
        mock_factory.create_provider.return_value = mock_provider

        await acompletion("provider/model/extra", messages=[{"role": "user", "content": "Hello"}])

        mock_provider.acompletion.assert_called_once()
        args, kwargs = mock_provider.acompletion.call_args
        assert isinstance(args[0], CompletionParams)
        assert args[0].model_id == "model/extra"
        assert args[0].messages == [{"role": "user", "content": "Hello"}]
        assert kwargs == {}


@pytest.mark.asyncio
async def test_completion_converts_chat_message_to_dict() -> None:
    mock_provider = Mock()
    mock_provider.acompletion = AsyncMock()

    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["provider"]
        mock_factory.get_provider_enum.return_value = ProviderName.OPENAI
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "gpt-4o")
        mock_factory.create_provider.return_value = mock_provider

        msg = ChatCompletionMessage(role="assistant", content="Hello", reasoning=Reasoning(content="Thinking..."))
        await acompletion("provider/gpt-4o", messages=[msg])

        mock_provider.acompletion.assert_called_once()
        args, _ = mock_provider.acompletion.call_args
        assert isinstance(args[0], CompletionParams)
        # reasoning shouldn't show up because it gets stripped out and only role and content are sent
        assert args[0].messages == [{"role": "assistant", "content": "Hello"}]


@pytest.mark.asyncio
async def test_all_providers_can_be_loaded(provider: str) -> None:
    """Test that all supported providers can be loaded successfully.

    This test uses the provider fixture which iterates over all providers
    returned by ProviderFactory.get_supported_providers(). It verifies that:
    1. Each provider can be imported and instantiated
    2. The created instance is actually a Provider
    3. No ImportError or other exceptions are raised during loading

    This test will automatically include new providers when they're added
    without requiring any code changes.
    """
    provider_instance = ProviderFactory.create_provider(provider, ClientConfig(api_key="test_key"))

    assert isinstance(provider_instance, Provider), f"Provider {provider} did not create a valid Provider instance"

    assert hasattr(provider_instance, "acompletion"), f"Provider {provider} does not have an acompletion method"
    assert callable(provider_instance.acompletion), f"Provider {provider} acompletion is not callable"


@pytest.mark.asyncio
async def test_all_providers_can_be_loaded_with_config(provider: str) -> None:
    """Test that all supported providers can be loaded with sample config parameters.

    This test verifies that providers can handle common configuration parameters
    like api_key and api_base without throwing errors during instantiation.
    """
    sample_config = ClientConfig(api_key="test_key", api_base="https://test.example.com")

    provider_instance = ProviderFactory.create_provider(provider, sample_config)

    assert isinstance(provider_instance, Provider), (
        f"Provider {provider} did not create a valid Provider instance with config"
    )


@pytest.mark.asyncio
async def test_provider_factory_can_create_all_supported_providers() -> None:
    """Test that ProviderFactory can create instances of all providers it claims to support."""
    supported_providers = ProviderFactory.get_supported_providers()

    for provider_name in supported_providers:
        provider_instance = ProviderFactory.create_provider(provider_name, ClientConfig(api_key="test_key"))

        assert isinstance(provider_instance, Provider), f"Failed to create valid Provider instance for {provider_name}"


@pytest.mark.asyncio
async def test_completion_empty_messages() -> None:
    """Test completion with empty messages list."""
    mock_provider = Mock()
    mock_provider.acompletion = AsyncMock()
    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["gemini"]
        mock_factory.get_provider_enum.return_value = ProviderName.OPENAI
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "gemini-pro")
        mock_factory.create_provider.return_value = mock_provider
        await acompletion("gemini-pro", messages=[])
        mock_provider.acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_completion_unsupported_image_format() -> None:
    """Test completion raises error for unsupported image format key."""
    with pytest.raises(ValueError, match="No valid image input found in message."):
        await acompletion(
            "gemini-pro", messages=[{"role": "user", "content": "Describe image", "image_jpeg": "not_supported"}]
        )


@pytest.mark.asyncio
async def test_completion_missing_role_field() -> None:
    """Test completion raises error for missing role field in message."""
    with pytest.raises(KeyError):
        await acompletion("gemini-pro", messages=[{"content": "Hello"}])


@pytest.mark.asyncio
async def test_completion_invalid_provider_name() -> None:
    """Test completion raises error for invalid provider name."""
    with pytest.raises(ValueError, match="Invalid provider name supplied"):
        await acompletion("invalid-provider/model", messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["provider:model", "provider/model"])
async def test_completion_valid_model_formats(model) -> None:
    """Test completion accepts both valid model formats."""
    mock_provider = Mock()
    mock_provider.acompletion = AsyncMock()
    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["provider"]
        mock_factory.get_provider_enum.return_value = ProviderName.OPENAI
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "model")
        mock_factory.create_provider.return_value = mock_provider
        await acompletion(model, messages=[{"role": "user", "content": "Hello"}])
        mock_provider.acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_completion_with_reasoning_field() -> None:
    """Test completion strips reasoning field from message before sending."""
    mock_provider = Mock()
    mock_provider.acompletion = AsyncMock()
    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["provider"]
        mock_factory.get_provider_enum.return_value = ProviderName.OPENAI
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "gpt-4o")
        mock_factory.create_provider.return_value = mock_provider
        msg = ChatCompletionMessage(role="user", content="Hello", reasoning=Reasoning(content="Thinking..."))
        await acompletion("provider/gpt-4o", messages=[msg])
        mock_provider.acompletion.assert_called_once()
        args, _ = mock_provider.acompletion.call_args
        assert isinstance(args[0], CompletionParams)
        assert args[0].messages == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_base64", [None, 123, "", "not_base64!", "===="])
async def test_completion_various_invalid_base64(invalid_base64) -> None:
    """Test completion raises error for various invalid base64 image data types."""
    with pytest.raises(ValueError):
        await acompletion(
            "gemini-pro", messages=[{"role": "user", "content": "Describe image", "image_base64": invalid_base64}]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_url", [None, 123, "", "ftp://invalid-url", "http:/bad"])
async def test_completion_various_invalid_image_url(invalid_url) -> None:
    """Test completion raises error for various invalid image URLs."""
    with pytest.raises(ValueError):
        await acompletion(
            "vertexai-pro", messages=[{"role": "user", "content": "Describe image", "image_url": invalid_url}]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_bytes", [None, "not_bytes", 123, [1, 2, 3]])
async def test_completion_various_invalid_image_bytes(invalid_bytes) -> None:
    """Test completion raises error for various non-bytes image_bytes."""
    with pytest.raises(ValueError):
        await acompletion(
            "gemini-pro", messages=[{"role": "user", "content": "Describe image", "image_bytes": invalid_bytes}]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4", "provider", "provider:", ":model", "provider:model:extra"])
async def test_completion_invalid_model_formats(model) -> None:
    """Test completion raises ValueError for various invalid model formats."""
    with pytest.raises(ValueError):
        await acompletion(model, messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_fields",
    [{"role": "user", "content": "Hello", "extra": 1}, {"role": "user", "content": "Hello", "reasoning": "why"}],
)
async def test_completion_message_with_extra_fields(extra_fields) -> None:
    """Test completion handles messages with extra fields."""
    mock_provider = Mock()
    mock_provider.acompletion = AsyncMock()
    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["provider"]
        mock_factory.get_provider_enum.return_value = ProviderName.OPENAI
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "gpt-4o")
        mock_factory.create_provider.return_value = mock_provider
        await acompletion("provider/gpt-4o", messages=[extra_fields])
        mock_provider.acompletion.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_name,config",
    [
        ("openai", ClientConfig(api_key="test_key")),
        ("vertexai", ClientConfig(api_key="test_key", api_base="https://vertex.example.com")),
        ("invalid", ClientConfig(api_key=None)),
    ],
)
async def test_provider_factory_various_configs(provider_name, config) -> None:
    """Test ProviderFactory with various provider names and configs."""
    if provider_name == "invalid":
        with pytest.raises(Exception):
            ProviderFactory.create_provider(provider_name, config)
    else:
        provider_instance = ProviderFactory.create_provider(provider_name, config)
        assert isinstance(provider_instance, Provider)
        assert hasattr(provider_instance, "acompletion")
        assert callable(provider_instance.acompletion)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_base64", [None, 123, "", "not_base64!", "====", [], {}, b"not_base64"])
async def test_completion_various_invalid_base64_final(invalid_base64) -> None:
    """Test completion raises error for final invalid base64 image data types."""
    with pytest.raises(ValueError):
        await acompletion(
            "gemini-pro", messages=[{"role": "user", "content": "Describe image", "image_base64": invalid_base64}]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_url", [None, 123, "", "ftp://invalid-url", "http:/bad", [], {}, b"not_url"])
async def test_completion_various_invalid_image_url_final(invalid_url) -> None:
    """Test completion raises error for final invalid image URLs."""
    with pytest.raises(ValueError):
        await acompletion(
            "vertexai-pro", messages=[{"role": "user", "content": "Describe image", "image_url": invalid_url}]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_bytes", [None, "not_bytes", 123, [1, 2, 3], {}, (), b"not_bytes"])
async def test_completion_various_invalid_image_bytes_final(invalid_bytes) -> None:
    """Test completion raises error for final non-bytes image_bytes."""
    with pytest.raises(ValueError):
        await acompletion(
            "gemini-pro", messages=[{"role": "user", "content": "Describe image", "image_bytes": invalid_bytes}]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "gpt-4",
        "provider",
        "provider:",
        ":model",
        "provider:model:extra",
        "provider/model/extra",
        "provider:model:extra:more",
        "provider:model:model",
    ],
)
async def test_completion_invalid_model_formats_final(model) -> None:
    """Test completion raises ValueError for final invalid model formats."""
    with pytest.raises(ValueError):
        await acompletion(model, messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_fields",
    [
        {"role": "user", "content": "Hello", "extra": 1},
        {"role": "user", "content": "Hello", "reasoning": "why"},
        {"role": "user", "content": "Hello", "unexpected": True},
        {"role": "user", "content": "Hello", "image_base64": None},
        {"role": "user", "content": "Hello", "image_url": None},
        {"role": "user", "content": "Hello", "image_bytes": None},
    ],
)
async def test_completion_message_with_final_extra_fields(extra_fields) -> None:
    """Test completion handles final messages with extra fields."""
    mock_provider = Mock()
    mock_provider.acompletion = AsyncMock()
    with patch("any_llm.utils.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["provider"]
        mock_factory.get_provider_enum.return_value = ProviderName.OPENAI
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "gpt-4o")
        mock_factory.create_provider.return_value = mock_provider
        await acompletion("provider/gpt-4o", messages=[extra_fields])
        mock_provider.acompletion.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_name,config",
    [
        ("openai", ClientConfig(api_key="test_key")),
        ("vertexai", ClientConfig(api_key="test_key", api_base="https://vertex.example.com")),
        ("invalid", ClientConfig(api_key=None)),
        ("openai", ClientConfig(api_key=None)),
        ("vertexai", ClientConfig(api_key=None)),
        ("openai", ClientConfig(api_key="")),
        ("vertexai", ClientConfig(api_key="")),
    ],
)
async def test_provider_factory_various_configs_final(provider_name, config) -> None:
    """Test ProviderFactory with final provider names and configs."""
    if provider_name == "invalid" or config.api_key is None or config.api_key == "":
        with pytest.raises(Exception):
            ProviderFactory.create_provider(provider_name, config)
    else:
        provider_instance = ProviderFactory.create_provider(provider_name, config)
        assert isinstance(provider_instance, Provider)
        assert hasattr(provider_instance, "acompletion")
        assert callable(provider_instance.acompletion)
