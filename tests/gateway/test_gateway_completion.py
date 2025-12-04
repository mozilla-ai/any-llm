import json
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from tests.gateway.conftest import MODEL_NAME

if TYPE_CHECKING:
    from tests.gateway.conftest import LiveServer


@pytest.mark.asyncio
async def test_chat_completion_with_provider_model_format(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
    test_messages: list[dict[str, str]],
) -> None:
    """Test chat completion using provider:model format."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": test_messages,
            "user": test_user["user_id"],
        },
        headers=api_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert "choices" in data
    assert data["choices"][0]["message"]["content"] is not None


@pytest.mark.asyncio
async def test_chat_completion_streaming(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
    test_messages: list[dict[str, str]],
) -> None:
    """Test streaming chat completion."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": test_messages,
            "user": test_user["user_id"],
            "stream": True,
        },
        headers=api_key_header,
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    chunks = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            chunk_data = json.loads(data_str)
            chunks.append(chunk_data)

    assert len(chunks) > 0
    assert "choices" in chunks[0]


@pytest.mark.asyncio
async def test_chat_completion_with_reasoning(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test that reasoning content is preserved in responses."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 2+2? Think step by step."}],
            "user": test_user["user_id"],
        },
        headers=api_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert "choices" in data
    message = data["choices"][0]["message"]
    assert "content" in message

    if message.get("reasoning"):
        assert isinstance(message["reasoning"], dict)
        assert "content" in message["reasoning"]
        assert len(message["reasoning"]["content"]) > 0


@pytest.mark.asyncio
async def test_chat_completion_with_temperature(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
    test_messages: list[dict[str, str]],
) -> None:
    """Test chat completion with temperature parameter."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": test_messages,
            "user": test_user["user_id"],
            "temperature": 0.7,
        },
        headers=api_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert "choices" in data
    assert data["choices"][0]["message"]["content"] is not None


@pytest.mark.asyncio
async def test_chat_completion_without_auth_header_fails(
    client: TestClient,
    test_messages: list[dict[str, str]],
) -> None:
    """Test that completion without authorization header fails."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": test_messages,
        },
    )

    assert response.status_code in [401, 422]


@pytest.mark.asyncio
async def test_chat_completion_with_invalid_api_key_fails(
    client: TestClient,
    test_messages: list[dict[str, str]],
) -> None:
    """Test that completion with invalid gateway API key fails."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": test_messages,
        },
        headers={"Authorization": "Bearer invalid-key"},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_completion_multi_turn_conversation(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test multi-turn conversation."""
    messages = [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What is my name?"},
    ]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "user": test_user["user_id"],
        },
        headers=api_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert "choices" in data
    content = data["choices"][0]["message"]["content"].lower()
    assert "alice" in content


def test_completion_with_openai_client(live_server: "LiveServer") -> None:
    """Test chat completion using the OpenAI SDK client with Authorization header."""
    from openai import OpenAI

    openai_client = OpenAI(
        base_url=f"{live_server.url}/v1",
        api_key=live_server.api_key,
    )
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Hello, world!"}],
    )

    assert response.choices[0].message.content is not None
