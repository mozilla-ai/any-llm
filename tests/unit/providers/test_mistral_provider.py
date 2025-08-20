from unittest.mock import Mock, patch

import httpx
import pytest

from any_llm.provider import ApiConfig
from any_llm.providers.mistral.utils import _patch_messages
from any_llm.types.completion import CompletionParams


def test_patch_messages_noop_when_no_tool_before_user() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    out = _patch_messages(messages)
    assert out == messages


def test_patch_messages_inserts_assistant_ok_between_tool_and_user() -> None:
    messages = [
        {"role": "tool", "content": "tool-output"},
        {"role": "user", "content": "next-question"},
    ]
    out = _patch_messages(messages)
    assert out == [
        {"role": "tool", "content": "tool-output"},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "next-question"},
    ]


def test_patch_messages_multiple_insertions() -> None:
    messages = [
        {"role": "tool", "content": "t1"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t2"},
        {"role": "user", "content": "u2"},
    ]
    out = _patch_messages(messages)
    assert out == [
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t2"},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "u2"},
    ]


def test_patch_messages_no_insertion_when_tool_at_end() -> None:
    messages = [
        {"role": "user", "content": "u"},
        {"role": "tool", "content": "t"},
    ]
    out = _patch_messages(messages)
    assert out == messages


def test_patch_messages_no_insertion_when_next_not_user() -> None:
    messages = [
        {"role": "tool", "content": "t"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u"},
    ]
    out = _patch_messages(messages)
    assert out == messages


def test_mistral_accepts_http_client() -> None:
    """Test that Mistral client accepts and passes through client parameter."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    api_key = "test-api-key"
    mock_http_client = Mock(spec=httpx.Client)

    with (
        patch("mistralai.Mistral") as mock_mistral,
        patch("any_llm.providers.mistral.utils._create_mistral_completion_from_response"),
        patch("any_llm.providers.mistral.utils._patch_messages", return_value=[]),
    ):
        mock_client = Mock()
        mock_mistral.return_value = mock_client
        mock_client.chat.complete.return_value = Mock()

        provider = MistralProvider(ApiConfig(api_key=api_key))
        provider.completion(
            CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]),
            client=mock_http_client,  # Note: Mistral uses 'client' not 'http_client'
        )

        # Verify Mistral client was instantiated with client parameter
        mock_mistral.assert_called_once_with(api_key=api_key, server_url=None, client=mock_http_client)


@pytest.mark.asyncio
async def test_mistral_accepts_http_client_async() -> None:
    """Test that Mistral client accepts and passes through client parameter in async."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    api_key = "test-api-key"
    mock_http_client = Mock(spec=httpx.AsyncClient)

    with (
        patch("mistralai.Mistral") as mock_mistral,
        patch("any_llm.providers.mistral.utils._create_mistral_completion_from_response"),
        patch("any_llm.providers.mistral.utils._patch_messages", return_value=[]),
    ):
        mock_client = Mock()
        mock_mistral.return_value = mock_client
        mock_client.chat.complete_async = Mock(return_value=Mock())

        provider = MistralProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(
            CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]),
            client=mock_http_client,  # Note: Mistral uses 'client' not 'http_client'
        )

        # Verify Mistral client was instantiated with client parameter
        mock_mistral.assert_called_once_with(api_key=api_key, server_url=None, client=mock_http_client)
