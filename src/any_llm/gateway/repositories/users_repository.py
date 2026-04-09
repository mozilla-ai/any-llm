from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.gateway.models.entities import User


async def get_active_user(db: AsyncSession, user_id: str, *, for_update: bool = False) -> User | None:
    """Query for a non-deleted user by user_id.

    Args:
        db: Database session
        user_id: User identifier
        for_update: If True, acquire a row-level lock (SELECT ... FOR UPDATE)

    Returns:
        User object if found and not soft-deleted, else None

    """
    stmt = select(User).where(User.user_id == user_id, User.deleted_at.is_(None))
    dialect = db.bind.dialect.name if db.bind else None
    if for_update and dialect != "sqlite":
        stmt = stmt.with_for_update()
    return (await db.execute(stmt)).scalar_one_or_none()
