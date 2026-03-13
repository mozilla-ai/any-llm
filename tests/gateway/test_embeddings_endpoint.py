"""Tests for the POST /v1/embeddings endpoint."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from any_llm.types.completion import CreateEmbeddingResponse, Embedding, Usage


def _mock_embedding_response() -> CreateEmbeddingResponse:
    """Build a real CreateEmbeddingResponse for testing."""
    return CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")],
        model="text-embedding-3-small",
        object="list",
        usage=Usage(prompt_tokens=10, total_tokens=10),
    )


def test_embeddings_requires_auth(client: TestClient) -> None:
    """POST /v1/embeddings requires authentication."""
    resp = client.post("/v1/embeddings", json={"model": "openai:text-embedding-3-small", "input": "hello"})
    assert resp.status_code == 401


def test_embeddings_with_api_key(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/embeddings works with API key authentication."""
    mock_resp = _mock_embedding_response()
    with patch("any_llm.gateway.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1


def test_embeddings_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """POST /v1/embeddings with master key requires 'user' field."""
    mock_resp = _mock_embedding_response()
    with patch("any_llm.gateway.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=master_key_header,
        )
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_embeddings_master_key_with_user(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """POST /v1/embeddings with master key + user field succeeds."""
    mock_resp = _mock_embedding_response()
    with patch("any_llm.gateway.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/embeddings",
            json={
                "model": "openai:text-embedding-3-small",
                "input": "hello",
                "user": test_user["user_id"],
            },
            headers=master_key_header,
        )
    assert resp.status_code == 200


def test_embeddings_list_input(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/embeddings accepts a list of strings as input."""
    mock_resp = _mock_embedding_response()
    with patch("any_llm.gateway.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": ["hello", "world"]},
            headers=api_key_header,
        )
    assert resp.status_code == 200


def test_embeddings_provider_error(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/embeddings returns 500 when the provider fails."""
    with patch(
        "any_llm.gateway.routes.embeddings.aembedding",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            "/v1/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 500
    assert "provider" in resp.json()["detail"].lower()


def test_embeddings_logs_usage(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/embeddings creates a usage log entry."""
    mock_resp = _mock_embedding_response()
    user_id = api_key_obj["user_id"]

    with patch("any_llm.gateway.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    embedding_logs = [log for log in logs if log["endpoint"] == "/v1/embeddings"]
    assert len(embedding_logs) >= 1
    assert embedding_logs[0]["status"] == "success"
    assert embedding_logs[0]["prompt_tokens"] == 10
    assert embedding_logs[0]["completion_tokens"] == 0


def test_embeddings_logs_error_on_failure(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/embeddings logs an error entry when the provider fails."""
    user_id = api_key_obj["user_id"]

    with patch(
        "any_llm.gateway.routes.embeddings.aembedding",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            "/v1/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 500

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    error_logs = [log for log in logs if log["endpoint"] == "/v1/embeddings" and log["status"] == "error"]
    assert len(error_logs) >= 1
    assert "provider down" in error_logs[0]["error_message"]


@pytest.mark.parametrize("extra_field", ["encoding_format", "dimensions"])
def test_embeddings_optional_fields(
    client: TestClient,
    api_key_header: dict[str, str],
    extra_field: str,
) -> None:
    """POST /v1/embeddings forwards optional OpenAI fields."""
    mock_resp = _mock_embedding_response()
    values = {"encoding_format": "float", "dimensions": 256}
    with patch("any_llm.gateway.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp) as mock:
        resp = client.post(
            "/v1/embeddings",
            json={
                "model": "openai:text-embedding-3-small",
                "input": "hello",
                extra_field: values[extra_field],
            },
            headers=api_key_header,
        )
    assert resp.status_code == 200
    call_kwargs = mock.call_args.kwargs
    assert extra_field in call_kwargs
    assert call_kwargs[extra_field] == values[extra_field]


def test_embeddings_cost_tracked_with_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/embeddings calculates cost when model pricing exists."""
    client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:text-embedding-3-small",
            "input_price_per_million": 0.02,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )

    mock_resp = _mock_embedding_response()
    user_id = api_key_obj["user_id"]

    with patch("any_llm.gateway.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    logs = usage_resp.json()
    embedding_logs = [log for log in logs if log["endpoint"] == "/v1/embeddings"]
    assert len(embedding_logs) >= 1
    assert embedding_logs[0]["cost"] is not None
    assert embedding_logs[0]["cost"] > 0
