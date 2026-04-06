from sqlalchemy.orm import Session

from any_llm.gateway.models.entities import User


def get_active_user(db: Session, user_id: str, *, for_update: bool = False) -> User | None:
    """Query for a non-deleted user by user_id.

    Args:
        db: Database session
        user_id: User identifier
        for_update: If True, acquire a row-level lock (SELECT ... FOR UPDATE)

    Returns:
        User object if found and not soft-deleted, else None

    """
    query = db.query(User).filter(User.user_id == user_id, User.deleted_at.is_(None))
    if for_update and (not db.bind or db.bind.dialect.name != "sqlite"):
        query = query.with_for_update()
    return query.first()
