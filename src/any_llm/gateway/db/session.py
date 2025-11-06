from collections.abc import Generator
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

_engine = None
_SessionLocal = None


def init_db(database_url: str, auto_migrate: bool = True) -> None:
    """Initialize database connection and optionally run migrations.

    Args:
        database_url: Database connection URL
        auto_migrate: If True, automatically run migrations to head. If False, skip migrations.
    """
    global _engine, _SessionLocal  # noqa: PLW0603

    _engine = create_engine(database_url, pool_pre_ping=True)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    if auto_migrate:
        alembic_cfg = Config()
        alembic_dir = Path(__file__).parent.parent / "alembic"
        alembic_cfg.set_main_option("script_location", str(alembic_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)

        command.upgrade(alembic_cfg, "head")


def get_db() -> Generator[Session]:
    """Get database session for dependency injection."""
    if _SessionLocal is None:
        msg = "Database not initialized. Call init_db() first."
        raise RuntimeError(msg)

    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
