"""Convert budget duration to seconds.

Revision ID: e7c85cc73bfa
Revises: 1e382aa3a9e7
Create Date: 2025-10-29 16:39:48.908608

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e7c85cc73bfa"
down_revision: str | Sequence[str] | None = "1e382aa3a9e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("budgets", sa.Column("budget_duration_sec", sa.Integer(), nullable=True))
    op.drop_column("budgets", "budget_duration")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column("budgets", sa.Column("budget_duration", sa.String(), nullable=True))
    op.drop_column("budgets", "budget_duration_sec")
