"""soft-delete user and cascade api key FK

Revision ID: a1b2c3d4e5f6
Revises: 5911f4bbf98d
Create Date: 2026-02-24 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "5911f4bbf98d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Naming convention so batch mode can find unnamed constraints on SQLite
naming_convention = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
}


def upgrade() -> None:
    """Add deleted_at column to users and set CASCADE on api_keys.user_id FK."""
    op.add_column("users", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))

    with op.batch_alter_table("api_keys", schema=None, naming_convention=naming_convention) as batch_op:
        batch_op.drop_constraint("fk_api_keys_user_id_users", type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_api_keys_user_id_users",
            "users",
            ["user_id"],
            ["user_id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    """Remove deleted_at column and restore original api_keys FK."""
    with op.batch_alter_table("api_keys", schema=None, naming_convention=naming_convention) as batch_op:
        batch_op.drop_constraint("fk_api_keys_user_id_users", type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_api_keys_user_id_users",
            "users",
            ["user_id"],
            ["user_id"],
        )

    op.drop_column("users", "deleted_at")
