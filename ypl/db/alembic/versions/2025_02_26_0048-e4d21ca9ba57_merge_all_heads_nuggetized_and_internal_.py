"""Merge all heads (nuggetized and internal only)

Revision ID: e4d21ca9ba57
Revises: 320fc8ee45ef, de9a8ad2284b
Create Date: 2025-02-26 00:48:54.101124+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'e4d21ca9ba57'
down_revision: str | None = ('320fc8ee45ef', 'de9a8ad2284b')
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
