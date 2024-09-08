"""add has_web column to user

Revision ID: f7e58d357687
Revises: bceb1e139447
Create Date: 2024-09-07 20:20:54.522620

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "f7e58d357687"
down_revision = "bceb1e139447"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user",
        sa.Column("has_web_login", sa.Boolean(), nullable=False, server_default="true"),
    )


def downgrade() -> None:
    op.drop_column("user", "has_web_login")
