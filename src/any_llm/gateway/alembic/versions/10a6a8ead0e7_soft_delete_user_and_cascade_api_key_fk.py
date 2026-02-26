"""soft-delete user and cascade api key FK

Revision ID: 10a6a8ead0e7
Revises: 5911f4bbf98d
Create Date: 2026-02-24 14:02:43.506983

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "10a6a8ead0e7"
down_revision: str | Sequence[str] | None = "5911f4bbf98d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add deleted_at column to users and set CASCADE on api_keys.user_id FK."""
    op.add_column("users", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_users_deleted_at", "users", ["deleted_at"])

    if op.get_bind().dialect.name != "sqlite":
        op.drop_constraint("api_keys_user_id_fkey", "api_keys", type_="foreignkey")
        op.create_foreign_key(
            "api_keys_user_id_fkey", "api_keys", "users", ["user_id"], ["user_id"], ondelete="CASCADE"
        )


def downgrade() -> None:
    """Remove deleted_at column and restore original api_keys FK."""
    if op.get_bind().dialect.name != "sqlite":
        op.drop_constraint("api_keys_user_id_fkey", "api_keys", type_="foreignkey")
        op.create_foreign_key("api_keys_user_id_fkey", "api_keys", "users", ["user_id"], ["user_id"])

    op.drop_index("ix_users_deleted_at", table_name="users")
    op.drop_column("users", "deleted_at")
