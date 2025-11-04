from collections.abc import Generator
from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from alembic import command

_engine = None
_SessionLocal = None


def init_db(database_url: str) -> None:
    """Initialize database connection and run migrations."""
    global _engine, _SessionLocal  # noqa: PLW0603

    _engine = create_engine(database_url, pool_pre_ping=True)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    alembic_cfg = Config()
    alembic_dir = Path(__file__).parent.parent.parent.parent.parent / "alembic"
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
