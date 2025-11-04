from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from any_llm.gateway.db.models import Base

_engine = None
_SessionLocal = None


def init_db(database_url: str) -> None:
    """Initialize database connection and create tables."""
    global _engine, _SessionLocal  # noqa: PLW0603

    _engine = create_engine(database_url, pool_pre_ping=True)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    Base.metadata.create_all(bind=_engine)


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
