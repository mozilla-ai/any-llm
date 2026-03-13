"""Tests for the GET /v1/models endpoint."""

from typing import Any

from fastapi.testclient import TestClient


def test_list_models_empty(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """GET /v1/models returns empty list when no pricing is configured."""
    resp = client.get("/v1/models", headers=master_key_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert data["data"] == []


def test_list_models_returns_configured_models(
    client: TestClient,
    master_key_header: dict[str, str],
    model_pricing: dict[str, Any],
) -> None:
    """GET /v1/models returns models from the pricing table."""
    resp = client.get("/v1/models", headers=master_key_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1

    model = data["data"][0]
    assert model["object"] == "model"
    assert "id" in model
    assert "created" in model
    assert isinstance(model["created"], int)
    assert "owned_by" in model


def test_list_models_owned_by_from_provider(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Models owned_by field is derived from the provider prefix."""
    client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )

    resp = client.get("/v1/models", headers=master_key_header)
    assert resp.status_code == 200
    models = resp.json()["data"]
    gpt4 = next(m for m in models if m["id"] == "openai:gpt-4o")
    assert gpt4["owned_by"] == "openai"


def test_get_model_found(
    client: TestClient,
    master_key_header: dict[str, str],
    model_pricing: dict[str, Any],
) -> None:
    """GET /v1/models/{model_id} returns the model when it exists."""
    model_key = model_pricing["model_key"]
    resp = client.get(f"/v1/models/{model_key}", headers=master_key_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == model_key
    assert data["object"] == "model"
    assert isinstance(data["created"], int)
    assert data["owned_by"] == model_key.split(":")[0]


def test_get_model_not_found(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """GET /v1/models/{model_id} returns 404 for unknown models."""
    resp = client.get("/v1/models/nonexistent:model", headers=master_key_header)
    assert resp.status_code == 404


def test_list_models_requires_auth(client: TestClient) -> None:
    """GET /v1/models requires authentication."""
    resp = client.get("/v1/models")
    assert resp.status_code == 401


def test_get_model_requires_auth(client: TestClient) -> None:
    """GET /v1/models/{model_id} requires authentication."""
    resp = client.get("/v1/models/openai:gpt-4o")
    assert resp.status_code == 401


def test_list_models_with_api_key(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """GET /v1/models works with API key authentication (not just master key)."""
    resp = client.get("/v1/models", headers=api_key_header)
    assert resp.status_code == 200
    assert resp.json()["object"] == "list"


def test_list_models_sorted_by_key(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """GET /v1/models returns models sorted by model_key."""
    for model_key in ["openai:gpt-4o", "anthropic:claude-3-haiku"]:
        client.post(
            "/v1/pricing",
            json={
                "model_key": model_key,
                "input_price_per_million": 1.0,
                "output_price_per_million": 1.0,
            },
            headers=master_key_header,
        )

    resp = client.get("/v1/models", headers=master_key_header)
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert ids == sorted(ids)
