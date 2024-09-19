"""set server default for language code

Revision ID: 8485cb1c12d0
Revises: e57e32b5aed7
Create Date: 2024-09-19 22:17:38.035325+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = '8485cb1c12d0'
down_revision: str | None = 'e57e32b5aed7'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("chat_messages", "language_code", server_default='EN')


def downgrade() -> None:
    op.alter_column("chat_messages", "language_code", server_default=None)
