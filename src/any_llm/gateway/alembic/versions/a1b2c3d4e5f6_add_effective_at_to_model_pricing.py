"""Add effective_at to model_pricing and change to composite PK.

Revision ID: a1b2c3d4e5f6
Revises: 967575f779b7
Create Date: 2026-04-04 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "967575f779b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add effective_at column and migrate to composite primary key."""
    # 1. Add nullable effective_at column
    op.add_column(
        "model_pricing",
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=True),
    )

    # 2. Backfill existing rows: effective_at = created_at
    op.execute("UPDATE model_pricing SET effective_at = created_at WHERE effective_at IS NULL")

    # 3. Make non-nullable
    op.alter_column("model_pricing", "effective_at", nullable=False)

    # 4. Drop old single-column PK, add composite PK
    op.drop_constraint("model_pricing_pkey", "model_pricing", type_="primary")
    op.create_primary_key("model_pricing_pkey", "model_pricing", ["model_key", "effective_at"])


def downgrade() -> None:
    """Revert to single-column primary key on model_key.

    Warning: this deletes all but the latest price entry per model.
    """
    # Keep only the latest entry per model_key
    op.execute("""
        DELETE FROM model_pricing AS mp
        WHERE mp.effective_at < (
            SELECT MAX(mp2.effective_at)
            FROM model_pricing AS mp2
            WHERE mp2.model_key = mp.model_key
        )
    """)

    op.drop_constraint("model_pricing_pkey", "model_pricing", type_="primary")
    op.create_primary_key("model_pricing_pkey", "model_pricing", ["model_key"])
    op.drop_column("model_pricing", "effective_at")
