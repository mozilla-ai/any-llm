from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
import ssl

from alembic import command
from alembic.config import Config
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

_engine: AsyncEngine | None = None
_SessionLocal: async_sessionmaker[AsyncSession] | None = None

# sslmode values that require TLS
SSL_MODES_REQUIRING_TLS = {"require", "verify-ca", "verify-full"}


def _to_async_url(database_url: str) -> tuple[str, dict[str, Any]]:
    """Translate a sync SQLAlchemy URL to its async-driver equivalent.

    Accepts URLs written against sync drivers (``postgresql://``, ``sqlite://``) or
    already-async URLs (``postgresql+asyncpg://``, ``sqlite+aiosqlite://``) and
    returns the async form.

    For asyncpg, strips ``sslmode`` from the query string (asyncpg doesn't
    accept it) and returns the equivalent ``connect_args`` instead.

    Returns:
        ``(url, connect_args)`` tuple.
    """
    connect_args: dict[str, Any] = {}

    if database_url.startswith("sqlite+aiosqlite://") or database_url.startswith("sqlite:///") or database_url.startswith("sqlite://"):
        if not database_url.startswith("sqlite+aiosqlite://"):
            database_url = "sqlite+aiosqlite://" + database_url[len("sqlite://") :]
        return database_url, connect_args

    # PostgreSQL — convert scheme and handle sslmode for asyncpg
    if database_url.startswith("postgresql+psycopg2://"):
        database_url = "postgresql+asyncpg://" + database_url[len("postgresql+psycopg2://") :]
    elif database_url.startswith("postgresql://"):
        database_url = "postgresql+asyncpg://" + database_url[len("postgresql://") :]

    # asyncpg doesn't accept sslmode as a query param — extract and convert
    parsed = urlparse(database_url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    sslmode_values = params.pop("sslmode", [])

    if sslmode_values:
        sslmode = sslmode_values[0]
        if sslmode in SSL_MODES_REQUIRING_TLS:
            if sslmode == "verify-full":
                connect_args["ssl"] = ssl.create_default_context()
            elif sslmode == "verify-ca":
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                connect_args["ssl"] = ctx
            else:
                # require — TLS without certificate verification
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                connect_args["ssl"] = ctx

        # Rebuild URL without sslmode
        remaining = urlencode(params, doseq=True)
        database_url = urlunparse(parsed._replace(query=remaining))

    return database_url, connect_args


def _make_async_engine(database_url: str, **extra: Any) -> AsyncEngine:
    """Create an asyncpg engine from a raw DATABASE_URL, handling sslmode."""
    async_url, connect_args = _to_async_url(database_url)
    kwargs: dict[str, Any] = {"pool_pre_ping": True, **extra}
    if connect_args:
        kwargs["connect_args"] = connect_args
    return create_async_engine(async_url, **kwargs)


def _to_sync_url(database_url: str) -> str:
    """Translate an async SQLAlchemy URL to its sync-driver equivalent, for Alembic."""
    if database_url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + database_url[len("postgresql+asyncpg://") :]
    if database_url.startswith("sqlite+aiosqlite://"):
        return "sqlite://" + database_url[len("sqlite+aiosqlite://") :]
    return database_url


def init_db(database_url: str, auto_migrate: bool = True) -> None:
    """Initialize the async database engine and optionally run migrations.

    Args:
        database_url: Database connection URL (sync or async form accepted)
        auto_migrate: If True, automatically run Alembic migrations to head.
    """
    global _engine, _SessionLocal  # noqa: PLW0603

    async_url, async_connect_args = _to_async_url(database_url)
    sync_url = _to_sync_url(database_url) if database_url != async_url else _to_sync_url(async_url)

    engine_kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if async_url.startswith("sqlite+aiosqlite://"):
        async_connect_args["check_same_thread"] = False
    if async_connect_args:
        engine_kwargs["connect_args"] = async_connect_args

    _engine = create_async_engine(async_url, **engine_kwargs)

    if _engine.dialect.name == "sqlite":
        sync_engine = _engine.sync_engine

        @event.listens_for(sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, _: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    _SessionLocal = async_sessionmaker(_engine, autocommit=False, autoflush=False, expire_on_commit=False)

    if auto_migrate:
        alembic_cfg = Config()
        alembic_dir = Path(__file__).parent.parent / "alembic"
        alembic_cfg.set_main_option("script_location", str(alembic_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", sync_url)

        command.upgrade(alembic_cfg, "head")


async def get_db() -> AsyncGenerator[AsyncSession]:
    """Get an async database session for FastAPI dependency injection."""
    if _SessionLocal is None:
        msg = "Database not initialized. Call init_db() first."
        raise RuntimeError(msg)

    async with _SessionLocal() as db:
        yield db


def create_session() -> AsyncSession:
    """Create a new async session, outside the FastAPI request scope.

    Callers are responsible for closing the session (use ``async with``).
    """
    if _SessionLocal is None:
        msg = "Database not initialized. Call init_db() first."
        raise RuntimeError(msg)
    return _SessionLocal()


async def reset_db() -> None:
    """Reset database state. Intended for testing only.

    Disposes the engine connection pool and clears the module-level references
    so that init_db() can be called again with different parameters.
    """
    global _engine, _SessionLocal  # noqa: PLW0603

    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _SessionLocal = None
