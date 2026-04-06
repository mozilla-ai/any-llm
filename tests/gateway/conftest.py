import os
import socket
import threading
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import uvicorn
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from testcontainers.postgres import PostgresContainer

from any_llm.gateway.core.config import API_KEY_HEADER, GatewayConfig
from any_llm.gateway.core.database import _make_async_engine
from any_llm.gateway.db import Base, get_db
from any_llm.gateway.main import create_app

MODEL_NAME = "gemini:gemini-2.5-flash"


def _run_alembic_migrations(database_url: str) -> None:
    """Run Alembic migrations for test database."""
    alembic_cfg = Config()
    alembic_dir = Path(__file__).parent.parent.parent / "src" / "any_llm" / "gateway" / "alembic"
    alembic_cfg.set_main_option("script_location", str(alembic_dir))
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    alembic_cfg.attributes["configure_logger"] = False
    command.upgrade(alembic_cfg, "head")


def _drop_all_sync(database_url: str) -> None:
    """Drop all tables (sync, uses psycopg2). Used for test teardown."""
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()
    finally:
        engine.dispose()


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


@pytest_asyncio.fixture
async def test_db(postgres_url: str) -> AsyncGenerator[AsyncSession]:
    """Create a test database session (async)."""
    _run_alembic_migrations(postgres_url)
    engine = _make_async_engine(postgres_url)
    session_factory = async_sessionmaker(engine, autocommit=False, autoflush=False, expire_on_commit=False)

    async with session_factory() as db:
        try:
            yield db
        finally:
            await db.close()

    await engine.dispose()
    _drop_all_sync(postgres_url)


@pytest_asyncio.fixture
async def db_session(test_config: GatewayConfig) -> AsyncGenerator[AsyncSession]:
    """Create a standalone DB session for verifying state outside the test client."""
    engine = _make_async_engine(test_config.database_url)
    session_factory = async_sessionmaker(engine, autocommit=False, autoflush=False, expire_on_commit=False)
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
    await engine.dispose()


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
    _run_alembic_migrations(test_config.database_url)
    engine = _make_async_engine(test_config.database_url)
    session_factory = async_sessionmaker(engine, autocommit=False, autoflush=False, expire_on_commit=False)
    app = create_app(test_config)

    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        _drop_all_sync(test_config.database_url)


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
