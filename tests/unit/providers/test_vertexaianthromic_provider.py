from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.vertexaianthropic.vertexaianthropic import VertexaianthropicProvider
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_vertexaianthropic_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.vertexaianthropic.vertexaianthropic.AnthropicVertex") as mock_vertex,
        patch("any_llm.providers.anthropic.base._convert_response"),
    ):
        mock_client = Mock()
        mock_vertex.return_value = mock_client
        mock_client.messages.create = AsyncMock()
        yield mock_vertex


@pytest.mark.asyncio
async def test_vertexaianthropic_client_created_with_project_id_and_region() -> None:
    project_id = "test-project"
    region = "us-east1"

    with mock_vertexaianthropic_provider() as mock_vertex:
        provider = VertexaianthropicProvider(project_id=project_id, region=region)
        await provider._acompletion(
            CompletionParams(model_id="claude-3-5-sonnet@20240620", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_vertex.assert_called_once_with(project_id=project_id, region=region)


@pytest.mark.asyncio
async def test_vertexaianthropic_client_uses_env_vars() -> None:
    with (
        mock_vertexaianthropic_provider() as mock_vertex,
        patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "env-project", "GOOGLE_CLOUD_LOCATION": "europe-west1"}),
    ):
        provider = VertexaianthropicProvider()
        await provider._acompletion(
            CompletionParams(model_id="claude-3-5-sonnet@20240620", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_vertex.assert_called_once_with(project_id="env-project", region="europe-west1")


@pytest.mark.asyncio
async def test_vertexaianthropic_client_defaults_region_to_us_central1() -> None:
    project_id = "test-project"

    with mock_vertexaianthropic_provider() as mock_vertex:
        provider = VertexaianthropicProvider(project_id=project_id)
        await provider._acompletion(
            CompletionParams(model_id="claude-3-5-sonnet@20240620", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_vertex.assert_called_once_with(project_id=project_id, region="us-central1")


def test_vertexaianthropic_raises_error_without_project_id() -> None:
    with (
        mock_vertexaianthropic_provider(),
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(MissingApiKeyError, match="GOOGLE_CLOUD_PROJECT"),
    ):
        VertexaianthropicProvider()


@pytest.mark.asyncio
async def test_vertexaianthropic_completion_calls_messages_create() -> None:
    project_id = "test-project"
    model = "claude-3-5-sonnet@20240620"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_vertexaianthropic_provider() as mock_vertex:
        provider = VertexaianthropicProvider(project_id=project_id)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        mock_vertex.return_value.messages.create.assert_called_once()
        call_kwargs = mock_vertex.return_value.messages.create.call_args[1]
        assert call_kwargs["model"] == model
        assert call_kwargs["messages"] == messages


@pytest.mark.asyncio
async def test_vertexaianthropic_completion_with_system_message() -> None:
    project_id = "test-project"
    model = "claude-3-5-sonnet@20240620"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    with mock_vertexaianthropic_provider() as mock_vertex:
        provider = VertexaianthropicProvider(project_id=project_id)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        call_kwargs = mock_vertex.return_value.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant."
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]


def test_vertexaianthropic_provider_name() -> None:
    with mock_vertexaianthropic_provider():
        provider = VertexaianthropicProvider(project_id="test-project")
        assert provider.PROVIDER_NAME == "vertexaianthropic"


def test_vertexaianthropic_env_api_key_name_empty() -> None:
    with mock_vertexaianthropic_provider():
        provider = VertexaianthropicProvider(project_id="test-project")
        assert provider.ENV_API_KEY_NAME == ""


def test_vertexaianthropic_does_not_support_list_models() -> None:
    with mock_vertexaianthropic_provider():
        provider = VertexaianthropicProvider(project_id="test-project")
        assert provider.SUPPORTS_LIST_MODELS is False


def test_vertexaianthropic_supports_completion() -> None:
    with mock_vertexaianthropic_provider():
        provider = VertexaianthropicProvider(project_id="test-project")
        assert provider.SUPPORTS_COMPLETION is True
        assert provider.SUPPORTS_COMPLETION_STREAMING is True
        assert provider.SUPPORTS_COMPLETION_REASONING is True
        assert provider.SUPPORTS_COMPLETION_IMAGE is True


@pytest.mark.asyncio
async def test_vertexaianthropic_constructor_arg_overrides_env_var() -> None:
    with (
        mock_vertexaianthropic_provider() as mock_vertex,
        patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "env-project", "GOOGLE_CLOUD_LOCATION": "europe-west1"}),
    ):
        provider = VertexaianthropicProvider(project_id="constructor-project", region="asia-east1")
        await provider._acompletion(
            CompletionParams(model_id="claude-3-5-sonnet@20240620", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_vertex.assert_called_once_with(project_id="constructor-project", region="asia-east1")
