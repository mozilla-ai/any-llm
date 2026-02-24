"""set null on user delete for log preservation

Revision ID: 5911f4bbf98d
Revises: e7c85cc73bfa
Create Date: 2026-02-24 13:24:13.439244

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5911f4bbf98d"
down_revision: str | Sequence[str] | None = "e7c85cc73bfa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Naming convention so batch mode can find unnamed constraints on SQLite
naming_convention = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
}


def upgrade() -> None:
    """Change FK constraints to SET NULL on delete for audit log preservation."""
    with op.batch_alter_table("usage_logs", schema=None, naming_convention=naming_convention) as batch_op:
        batch_op.drop_constraint("fk_usage_logs_user_id_users", type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_usage_logs_user_id_users",
            "users",
            ["user_id"],
            ["user_id"],
            ondelete="SET NULL",
        )
        batch_op.drop_constraint("fk_usage_logs_api_key_id_api_keys", type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_usage_logs_api_key_id_api_keys",
            "api_keys",
            ["api_key_id"],
            ["id"],
            ondelete="SET NULL",
        )

    with op.batch_alter_table("budget_reset_logs", schema=None, naming_convention=naming_convention) as batch_op:
        batch_op.alter_column("user_id", existing_type=sa.String(), nullable=True)
        batch_op.drop_constraint("fk_budget_reset_logs_user_id_users", type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_budget_reset_logs_user_id_users",
            "users",
            ["user_id"],
            ["user_id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    """Restore original FK constraints without ON DELETE SET NULL."""
    # Remove orphaned rows before restoring NOT NULL constraint
    op.execute(sa.text("DELETE FROM budget_reset_logs WHERE user_id IS NULL"))

    with op.batch_alter_table("budget_reset_logs", schema=None, naming_convention=naming_convention) as batch_op:
        batch_op.drop_constraint("fk_budget_reset_logs_user_id_users", type_="foreignkey")
        batch_op.create_foreign_key("fk_budget_reset_logs_user_id_users", "users", ["user_id"], ["user_id"])
        batch_op.alter_column("user_id", existing_type=sa.String(), nullable=False)

    with op.batch_alter_table("usage_logs", schema=None, naming_convention=naming_convention) as batch_op:
        batch_op.drop_constraint("fk_usage_logs_api_key_id_api_keys", type_="foreignkey")
        batch_op.create_foreign_key("fk_usage_logs_api_key_id_api_keys", "api_keys", ["api_key_id"], ["id"])
        batch_op.drop_constraint("fk_usage_logs_user_id_users", type_="foreignkey")
        batch_op.create_foreign_key("fk_usage_logs_user_id_users", "users", ["user_id"], ["user_id"])
