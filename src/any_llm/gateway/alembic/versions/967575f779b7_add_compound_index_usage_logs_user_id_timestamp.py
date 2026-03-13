"""add compound index on usage_logs(user_id, timestamp)

Revision ID: 967575f779b7
Revises: 10a6a8ead0e7
Create Date: 2026-03-13 14:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "967575f779b7"
down_revision: str | Sequence[str] | None = "10a6a8ead0e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add compound index on (user_id, timestamp DESC) for usage queries."""
    op.create_index(
        "ix_usage_logs_user_id_timestamp",
        "usage_logs",
        ["user_id", "timestamp"],
    )


def downgrade() -> None:
    """Remove compound index."""
    op.drop_index("ix_usage_logs_user_id_timestamp", table_name="usage_logs")
