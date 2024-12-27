"""add prompt modifier description

Revision ID: 404e97438b5e
Revises: f846bd03e883
Create Date: 2024-12-27 07:33:56.577576+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = '404e97438b5e'
down_revision: str | None = 'f846bd03e883'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('prompt_modifiers', sa.Column('description', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('prompt_modifiers', 'description')
    # ### end Alembic commands ###
