"""enable contextual retrieval

Revision ID: 8e1ac4f39a9f
Revises: b7a7eee5aa15
Create Date: 2024-12-20 13:29:09.918661

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "8e1ac4f39a9f"
down_revision = "b7a7eee5aa15"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "search_settings",
        sa.Column(
            "enable_contextual_rag",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )


def downgrade() -> None:
    op.drop_column("search_settings", "enable_contextual_rag")
