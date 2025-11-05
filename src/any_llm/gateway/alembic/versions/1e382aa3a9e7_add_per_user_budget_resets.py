"""Add per-user budget resets.

Revision ID: 1e382aa3a9e7
Revises: 28d153c22616
Create Date: 2025-10-29 16:25:20.350313

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1e382aa3a9e7"
down_revision: str | Sequence[str] | None = "28d153c22616"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("budgets", "budget_reset_at")

    op.add_column("users", sa.Column("budget_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("next_budget_reset_at", sa.DateTime(timezone=True), nullable=True))

    op.create_table(
        "budget_reset_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("budget_id", sa.String(), nullable=False),
        sa.Column("previous_spend", sa.Float(), nullable=False),
        sa.Column("reset_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("next_reset_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
        ),
        sa.ForeignKeyConstraint(
            ["budget_id"],
            ["budgets.budget_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_budget_reset_logs_user_id"), "budget_reset_logs", ["user_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_budget_reset_logs_user_id"), table_name="budget_reset_logs")
    op.drop_table("budget_reset_logs")

    op.drop_column("users", "next_budget_reset_at")
    op.drop_column("users", "budget_started_at")

    op.add_column("budgets", sa.Column("budget_reset_at", sa.DateTime(timezone=True), nullable=True))
