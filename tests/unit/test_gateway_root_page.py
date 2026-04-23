from pathlib import Path

from fastapi.testclient import TestClient

from any_llm.gateway.core.config import GatewayConfig
from any_llm.gateway.main import create_app


def test_root_tutorial_page_is_available(tmp_path: Path) -> None:
    database_path = tmp_path / "gateway-root-test.db"
    config = GatewayConfig(database_url=f"sqlite:///{database_path}")
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI Gateway (Proxy Server)" in response.text
    assert "Gateway Quickstart" in response.text
    assert "bootstrap API key" in response.text
    assert "from openai import OpenAI" in response.text
    assert "YOUR_GATEWAY_KEY" in response.text
    assert "docs.mozilla.ai/any-llm/gateway/quickstart" in response.text
