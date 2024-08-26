"""make is_public non null

Revision ID: c46f020ab21c
Revises: bbb0c0af1ace
Create Date: 2024-08-26 22:37:36.103158+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'c46f020ab21c'
down_revision: str | None = 'bbb0c0af1ace'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('chats', 'is_public',
               existing_type=sa.BOOLEAN(),
               nullable=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('chats', 'is_public',
               existing_type=sa.BOOLEAN(),
               nullable=True)
    # ### end Alembic commands ###
