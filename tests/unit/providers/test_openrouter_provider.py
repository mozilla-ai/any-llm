from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.providers.openrouter import OpenrouterProvider
from any_llm.providers.openrouter.utils import _convert_models_list
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionParams,
    Reasoning,
)
from any_llm.types.model import Model


@pytest.mark.asyncio
async def test_reasoning_param_added_with_explicit_effort() -> None:
    """Test that reasoning parameter is added when reasoning_effort is specified."""
    mock_completion = ChatCompletion(
        id="test-123",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Hello!",
                    reasoning=Reasoning(content="Let me think..."),
                ),
            )
        ],
    )

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="low",
        )

        provider = OpenrouterProvider(api_key="sk-test")
        await provider._acompletion(params)

        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "low"


@pytest.mark.asyncio
async def test_reasoning_auto_excludes_reasoning() -> None:
    """Test that reasoning_effort='auto' does not include reasoning - no extra_body at all."""
    mock_completion = ChatCompletion(
        id="test-456",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="Hello!"),
            )
        ],
    )

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        provider = OpenrouterProvider(api_key="sk-test")
        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        await provider._acompletion(params)

        call_args = mock_client.chat.completions.create.call_args
        # Should not have extra_body at all when "auto"
        assert "extra_body" not in call_args.kwargs


@pytest.mark.asyncio
async def test_reasoning_none_excludes_reasoning() -> None:
    """Test that reasoning_effort='none' does not include reasoning - no extra_body at all."""
    mock_completion = ChatCompletion(
        id="test-none",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="Hello!"),
            )
        ],
    )

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        provider = OpenrouterProvider(api_key="sk-test")
        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="none",
        )

        await provider._acompletion(params)

        call_args = mock_client.chat.completions.create.call_args
        # Should not have extra_body at all when "none"
        assert "extra_body" not in call_args.kwargs


@pytest.mark.asyncio
async def test_reasoning_with_custom_reasoning_object() -> None:
    """Test that custom reasoning object overrides reasoning_effort."""
    mock_completion = ChatCompletion(
        id="test-custom",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Hello!",
                    reasoning=Reasoning(content="Custom reasoning"),
                ),
            )
        ],
    )

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="low",  # This should be overridden
        )

        provider = OpenrouterProvider(api_key="sk-test")
        await provider._acompletion(params, reasoning={"effort": "high", "max_tokens": 1000})

        call_args = mock_client.chat.completions.create.call_args
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "high"
        assert call_args.kwargs["extra_body"]["reasoning"]["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_no_extra_body_when_no_reasoning() -> None:
    """Test that no extra_body is sent when reasoning is not requested."""
    mock_completion = ChatCompletion(
        id="test-no-extra",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="Hello!"),
            )
        ],
    )

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort=None,  # No reasoning
        )
        provider = OpenrouterProvider(api_key="sk-test")
        await provider._acompletion(params)

        call_args = mock_client.chat.completions.create.call_args
        # Should not have extra_body at all when no reasoning
        assert "extra_body" not in call_args.kwargs


@pytest.mark.asyncio
async def test_preserve_existing_extra_body() -> None:
    """Test that existing extra_body is preserved when no reasoning is added."""
    mock_completion = ChatCompletion(
        id="test-preserve",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="Hello!"),
            )
        ],
    )

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        provider = OpenrouterProvider(api_key="sk-test")
        await provider._acompletion(params, extra_body={"some_other_param": "value"})

        call_args = mock_client.chat.completions.create.call_args
        # Should preserve existing extra_body but not add reasoning
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["some_other_param"] == "value"
        assert "reasoning" not in call_args.kwargs["extra_body"]


@pytest.mark.asyncio
async def test_streaming_with_reasoning() -> None:
    """Test that streaming passes through reasoning parameters correctly."""
    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        params = CompletionParams(
            model_id="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="high",
            stream=True,
        )
        provider = OpenrouterProvider(api_key="sk-test")
        await provider._acompletion(params)

        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["stream"] is True
        assert "extra_body" in call_args.kwargs
        assert call_args.kwargs["extra_body"]["reasoning"]["effort"] == "high"


def test_openrouter_remaps_max_tokens_to_max_completion_tokens() -> None:
    params = CompletionParams(model_id="openai/gpt-4", messages=[{"role": "user", "content": "Hello"}], max_tokens=8192)
    result = OpenrouterProvider._convert_completion_params(params)
    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 8192


def _make_openrouter_model(**overrides: object) -> Model:
    """Build a fake model object resembling what the OpenAI SDK constructs from an OpenRouter response.

    Uses ``Model.model_construct`` to match how the OpenAI SDK actually builds
    model objects from raw API responses, including extra OpenRouter fields.
    """
    defaults: dict[str, object] = {
        "id": "qwen/qwen3.7-max",
        "created": 1779376861,
        "object": None,
        "owned_by": None,
    }
    defaults.update(overrides)
    return Model.model_construct(**defaults)  # type: ignore[arg-type]


def test_convert_models_list_fills_missing_object_and_owned_by() -> None:
    """Models with None object/owned_by are rebuilt with valid defaults."""
    response = SimpleNamespace(data=[_make_openrouter_model()])
    result = _convert_models_list(response)

    assert len(result) == 1
    model = result[0]
    assert model.id == "qwen/qwen3.7-max"
    assert model.object == "model"
    assert model.owned_by == "openrouter"
    assert model.created == 1779376861


def test_convert_models_list_preserves_existing_owned_by() -> None:
    """When the API does provide owned_by, it should be kept."""
    response = SimpleNamespace(data=[_make_openrouter_model(owned_by="qwen")])
    result = _convert_models_list(response)

    assert result[0].owned_by == "qwen"


def test_convert_models_list_defaults_created_to_zero_when_none() -> None:
    """A None created timestamp falls back to 0."""
    response = SimpleNamespace(data=[_make_openrouter_model(created=None)])
    result = _convert_models_list(response)

    assert result[0].created == 0


def test_convert_models_list_handles_empty_response() -> None:
    response = SimpleNamespace(data=[])
    result = _convert_models_list(response)

    assert result == []


def test_convert_models_list_round_trip_serialization() -> None:
    """Converted models must survive model_dump_json -> model_validate_json."""
    response = SimpleNamespace(data=[_make_openrouter_model()])
    models = _convert_models_list(response)

    for model in models:
        json_data = model.model_dump_json()
        restored = Model.model_validate_json(json_data)
        assert restored.id == model.id
        assert restored.object == "model"
        assert restored.owned_by == "openrouter"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_list_models_returns_valid_model_objects(mock_openai_class: MagicMock) -> None:
    """End-to-end: OpenrouterProvider.list_models() returns spec-compliant Model objects."""
    raw_models = [
        _make_openrouter_model(id="openai/gpt-4o"),
        _make_openrouter_model(id="anthropic/claude-3.5-sonnet", owned_by="anthropic"),
    ]
    mock_client = AsyncMock()
    mock_client.models.list.return_value = SimpleNamespace(data=raw_models)
    mock_openai_class.return_value = mock_client

    provider = OpenrouterProvider(api_key="sk-test")
    result = provider.list_models()

    assert len(result) == 2
    for model in result:
        assert isinstance(model, Model)
        assert model.object == "model"
        assert model.owned_by is not None

    assert result[0].id == "openai/gpt-4o"
    assert result[0].owned_by == "openrouter"
    assert result[1].id == "anthropic/claude-3.5-sonnet"
    assert result[1].owned_by == "anthropic"


def test_convert_models_list_preserves_extra_openrouter_fields() -> None:
    """Extra OpenRouter attributes (name, pricing, etc.) survive conversion."""
    model = _make_openrouter_model(
        name="Qwen3.7 Max",
        pricing={"prompt": "0.001", "completion": "0.002"},
    )
    response = SimpleNamespace(data=[model])
    result = _convert_models_list(response)

    assert len(result) == 1
    converted = result[0]
    assert converted.id == "qwen/qwen3.7-max"
    assert converted.object == "model"
    assert converted.owned_by == "openrouter"
    assert converted.model_extra is not None
    assert converted.model_extra["name"] == "Qwen3.7 Max"
    assert converted.model_extra["pricing"] == {"prompt": "0.001", "completion": "0.002"}
