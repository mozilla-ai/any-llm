from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from any_llm.provider import ApiConfig
from any_llm.providers.watsonx.watsonx import WatsonxProvider


@contextmanager
def mock_watsonx_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.watsonx.watsonx.ModelInference") as mock_model_inference,
        patch("any_llm.providers.watsonx.watsonx._convert_response") as mock_convert_response,
    ):
        mock_model_instance = MagicMock()
        mock_model_inference.return_value = mock_model_instance

        mock_watsonx_response = {"choices": [{"message": {"content": "Hello"}}]}
        mock_model_instance.chat.return_value = mock_watsonx_response

        mock_openai_response = MagicMock()
        mock_convert_response.return_value = mock_openai_response

        yield mock_model_instance, mock_convert_response, mock_model_inference


@contextmanager
def mock_watsonx_streaming_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.watsonx.watsonx.ModelInference") as mock_model_inference,
        patch("any_llm.providers.watsonx.watsonx._convert_streaming_chunk") as mock_convert_streaming_chunk,
    ):
        mock_model_instance = MagicMock()
        mock_model_inference.return_value = mock_model_instance

        # Mock the streaming response
        mock_watsonx_chunk1 = {"choices": [{"delta": {"content": "Hello"}}]}
        mock_watsonx_chunk2 = {"choices": [{"delta": {"content": " World"}}]}
        mock_model_instance.chat_stream.return_value = [mock_watsonx_chunk1, mock_watsonx_chunk2]

        mock_openai_chunk1 = MagicMock()
        mock_openai_chunk2 = MagicMock()
        mock_convert_streaming_chunk.side_effect = [mock_openai_chunk1, mock_openai_chunk2]

        yield mock_model_instance, mock_convert_streaming_chunk, mock_model_inference


def test_watsonx_non_streaming() -> None:
    api_key = "test-api-key"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_watsonx_provider() as (mock_model_instance, mock_convert_response, mock_model_inference):
        provider = WatsonxProvider(ApiConfig(api_key=api_key))
        result = provider._make_api_call("test-model", messages)

        # Verify ModelInference was created with correct parameters
        mock_model_inference.assert_called_once()
        call_kwargs = mock_model_inference.call_args[1]
        assert call_kwargs["model_id"] == "test-model"

        # Verify chat was called with correct parameters
        mock_model_instance.chat.assert_called_once_with(messages=messages, params={})

        # Verify response conversion was called
        mock_convert_response.assert_called_once()

        assert result == mock_convert_response.return_value


def test_watsonx_streaming() -> None:
    api_key = "test-api-key"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_watsonx_streaming_provider() as (mock_model_instance, mock_convert_streaming_chunk, mock_model_inference):
        provider = WatsonxProvider(ApiConfig(api_key=api_key))
        result = provider._make_api_call("test-model", messages, stream=True)

        mock_model_inference.assert_called_once()
        call_kwargs = mock_model_inference.call_args[1]
        assert call_kwargs["model_id"] == "test-model"

        result_list = list(result)

        mock_model_instance.chat_stream.assert_called_once_with(messages=messages, params={"stream": True})

        assert mock_convert_streaming_chunk.call_count == 2

        assert len(result_list) == 2
        assert result_list is not None


def test_watsonx_supports_streaming() -> None:
    """Test that WatsonxProvider correctly advertises streaming support."""
    provider = WatsonxProvider(ApiConfig(api_key="test-key"))
    assert provider.SUPPORTS_STREAMING is True


def test_watsonx_verify_kwargs() -> None:
    """Test that verify_kwargs doesn't raise any errors."""
    provider = WatsonxProvider(ApiConfig(api_key="test-key"))
    provider.verify_kwargs({"stream": True})
    provider.verify_kwargs({"stream": False})
    provider.verify_kwargs({})
