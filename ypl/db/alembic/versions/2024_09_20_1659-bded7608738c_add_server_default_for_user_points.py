"""Add server_default for user points.

Revision ID: bded7608738c
Revises: 092b7baae511
Create Date: 2024-09-20 16:59:54.441026+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'bded7608738c'
down_revision: str | None = '092b7baae511'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("users", "points", server_default='10000')


def downgrade() -> None:
    op.alter_column("users", "points", server_default=None)
