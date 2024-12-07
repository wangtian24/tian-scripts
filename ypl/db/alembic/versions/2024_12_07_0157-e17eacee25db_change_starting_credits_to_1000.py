"""change starting credits to 1000

Revision ID: e17eacee25db
Revises: cf08324b0b90
Create Date: 2024-12-07 01:57:43.814194+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'e17eacee25db'
down_revision: str | None = 'cf08324b0b90'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column('users', 'points', server_default="1000")


def downgrade() -> None:
    op.alter_column('users', 'points', server_default="500")
