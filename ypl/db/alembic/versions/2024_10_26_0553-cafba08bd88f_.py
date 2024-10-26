"""empty message

Revision ID: cafba08bd88f
Revises: 4a35af8edef7, 3dc4fad174d4
Create Date: 2024-10-26 05:53:41.791450+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'cafba08bd88f'
down_revision: str | None = ('4a35af8edef7', '3dc4fad174d4')
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
