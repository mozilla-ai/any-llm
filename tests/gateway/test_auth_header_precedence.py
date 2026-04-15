"""End-to-end tests for the canonical ``AnyLLM-Key`` authentication header.

Precedence and malformed-input branches are unit-tested in
``tests/unit/test_extract_bearer_token.py``; this module focuses on behaviour that
only emerges through the full middleware stack: CORS allow-listing, the ``Vary``
caching hint, and the virtual-API-key auth path.
"""

from fastapi.testclient import TestClient

from any_llm.gateway.core.config import API_KEY_HEADER, GatewayConfig


def test_canonical_header_constant_is_rfc6648_compliant() -> None:
    """The auth header constant must be the RFC 6648-compliant name (no ``X-`` prefix)."""
    assert API_KEY_HEADER == "AnyLLM-Key"


def test_canonical_header_accepted(client: TestClient, test_config: GatewayConfig) -> None:
    """Requests using the canonical header authenticate against the master-key path."""
    response = client.post(
        "/v1/keys",
        json={"key_name": "canonical-header-test"},
        headers={API_KEY_HEADER: f"Bearer {test_config.master_key}"},
    )

    assert response.status_code == 200
    assert response.json()["key_name"] == "canonical-header-test"


def test_virtual_api_key_with_canonical_header(client: TestClient, api_key_header: dict[str, str]) -> None:
    """A virtual API key authenticates on the canonical header for the models endpoint."""
    response = client.get("/v1/models", headers=api_key_header)

    assert response.status_code == 200


def test_authenticated_response_vary_includes_canonical_header(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Authenticated responses advertise Vary on both ``Authorization`` and ``AnyLLM-Key``."""
    response = client.get("/v1/keys", headers=master_key_header)

    assert response.status_code == 200
    vary = response.headers.get("Vary", "")
    assert "Authorization" in vary
    assert API_KEY_HEADER in vary


def test_canonical_header_added_to_cors_allowlist(postgres_url: str) -> None:
    """The canonical ``AnyLLM-Key`` header must be included in the CORS allow-list."""
    from any_llm.gateway.main import create_app

    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        cors_allow_origins=["https://trusted.com"],
    )
    app = create_app(config)

    with TestClient(app) as test_client:
        response = test_client.options(
            "/v1/keys",
            headers={
                "Origin": "https://trusted.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": API_KEY_HEADER,
            },
        )
        assert response.status_code == 200
        allowed = response.headers.get("access-control-allow-headers", "").lower()
        assert API_KEY_HEADER.lower() in allowed
