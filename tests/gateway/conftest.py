import os
import socket
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

from any_llm.gateway.config import API_KEY_HEADER, GatewayConfig
from any_llm.gateway.db import Base, get_db
from any_llm.gateway.server import create_app

MODEL_NAME = "gemini:gemini-2.5-flash"


def _run_alembic_migrations(database_url: str) -> None:
    """Run Alembic migrations for test database."""
    alembic_cfg = Config()
    alembic_dir = Path(__file__).parent.parent.parent / "src" / "any_llm" / "gateway" / "alembic"
    alembic_cfg.set_main_option("script_location", str(alembic_dir))
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    alembic_cfg.attributes["configure_logger"] = False
    command.upgrade(alembic_cfg, "head")


@pytest.fixture(scope="session")
def postgres_url() -> Generator[str]:
    """Get PostgreSQL URL from environment or start temporary container."""
    if url := os.getenv("TEST_DATABASE_URL"):
        yield url
    else:
        postgres = PostgresContainer("postgres:17", username="test", password="test", dbname="test_db")  # noqa: S106
        postgres.start()
        try:
            yield postgres.get_connection_url()
        finally:
            postgres.stop()


@pytest.fixture
def test_db(postgres_url: str) -> Generator[Session]:
    """Create a test database session."""
    engine = create_engine(postgres_url, pool_pre_ping=True)
    _run_alembic_migrations(postgres_url)

    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = testing_session_local()

    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()


@pytest.fixture(scope="session")
def test_config(postgres_url: str) -> GatewayConfig:
    """Create a test configuration."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
    )


@pytest.fixture
def client(test_config: GatewayConfig) -> Generator[TestClient]:
    """Create a test client for the FastAPI app."""
    from sqlalchemy import text

    _run_alembic_migrations(test_config.database_url)
    engine = create_engine(test_config.database_url, pool_pre_ping=True)
    app = create_app(test_config)

    def override_get_db() -> Generator[Session]:
        testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()


@pytest.fixture
def master_key_header(test_config: GatewayConfig) -> dict[str, str]:
    """Return authentication header with master key."""
    header_name = API_KEY_HEADER
    return {header_name: f"Bearer {test_config.master_key}"}


@pytest.fixture
def api_key_obj(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    """Create a test API key and return its details."""
    response = client.post(
        "/v1/keys",
        json={"key_name": "test-key"},
        headers=master_key_header,
    )
    assert response.status_code == 200
    result: dict[str, Any] = response.json()
    return result


@pytest.fixture
def api_key_header(test_config: GatewayConfig, api_key_obj: dict[str, Any]) -> dict[str, str]:
    """Return authentication header with API key."""
    header_name = API_KEY_HEADER
    return {header_name: f"Bearer {api_key_obj['key']}"}


@pytest.fixture
def test_user(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    """Create a test user."""
    response = client.post(
        "/v1/users",
        json={"user_id": "test-user", "alias": "Test User"},
        headers=master_key_header,
    )
    assert response.status_code == 200
    result: dict[str, Any] = response.json()
    return result


@pytest.fixture
def test_messages() -> list[dict[str, str]]:
    """Return test messages for completion requests."""
    return [{"role": "user", "content": "Say 'hello' and nothing else"}]


@pytest.fixture
def test_messages_with_longer_response() -> list[dict[str, str]]:
    """Return test messages for completion requests with usage."""
    return [{"role": "user", "content": "Tell me a brief story"}]


@pytest.fixture
def model_pricing(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    """Create model pricing for gemini-2.5-flash."""
    response = client.post(
        "/v1/pricing",
        json={
            "model_key": MODEL_NAME,
            "input_price_per_million": 0.075,
            "output_price_per_million": 0.30,
        },
        headers=master_key_header,
    )
    assert response.status_code == 200
    result: dict[str, Any] = response.json()
    return result


@dataclass
class LiveServer:
    """Holds information about a running test server."""

    url: str
    api_key: str


@pytest.fixture
def live_server(test_config: GatewayConfig, api_key_obj: dict[str, Any]) -> Generator[LiveServer]:
    """Start a live uvicorn server and yield its URL and API key."""
    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    app = create_app(test_config)

    server_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(server_config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)

    try:
        yield LiveServer(url=f"http://127.0.0.1:{port}", api_key=api_key_obj["key"])
    finally:
        server.should_exit = True
        thread.join(timeout=5)
