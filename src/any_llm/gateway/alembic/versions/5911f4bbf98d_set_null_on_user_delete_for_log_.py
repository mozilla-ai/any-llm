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


def upgrade() -> None:
    """Change FK constraints to SET NULL on delete for audit log preservation."""
    if op.get_bind().dialect.name == "sqlite":
        return

    op.drop_constraint("usage_logs_user_id_fkey", "usage_logs", type_="foreignkey")
    op.create_foreign_key(
        "usage_logs_user_id_fkey", "usage_logs", "users", ["user_id"], ["user_id"], ondelete="SET NULL"
    )

    op.drop_constraint("usage_logs_api_key_id_fkey", "usage_logs", type_="foreignkey")
    op.create_foreign_key(
        "usage_logs_api_key_id_fkey", "usage_logs", "api_keys", ["api_key_id"], ["id"], ondelete="SET NULL"
    )

    op.alter_column("budget_reset_logs", "user_id", existing_type=sa.String(), nullable=True)
    op.drop_constraint("budget_reset_logs_user_id_fkey", "budget_reset_logs", type_="foreignkey")
    op.create_foreign_key(
        "budget_reset_logs_user_id_fkey", "budget_reset_logs", "users", ["user_id"], ["user_id"], ondelete="SET NULL"
    )


def downgrade() -> None:
    """Restore original FK constraints without ON DELETE SET NULL."""
    if op.get_bind().dialect.name == "sqlite":
        return

    op.execute(sa.text("DELETE FROM budget_reset_logs WHERE user_id IS NULL"))

    op.drop_constraint("budget_reset_logs_user_id_fkey", "budget_reset_logs", type_="foreignkey")
    op.create_foreign_key("budget_reset_logs_user_id_fkey", "budget_reset_logs", "users", ["user_id"], ["user_id"])
    op.alter_column("budget_reset_logs", "user_id", existing_type=sa.String(), nullable=False)

    op.drop_constraint("usage_logs_api_key_id_fkey", "usage_logs", type_="foreignkey")
    op.create_foreign_key("usage_logs_api_key_id_fkey", "usage_logs", "api_keys", ["api_key_id"], ["id"])

    op.drop_constraint("usage_logs_user_id_fkey", "usage_logs", type_="foreignkey")
    op.create_foreign_key("usage_logs_user_id_fkey", "usage_logs", "users", ["user_id"], ["user_id"])
