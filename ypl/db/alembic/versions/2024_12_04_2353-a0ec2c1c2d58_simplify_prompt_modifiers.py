"""simplify prompt modifiers

Revision ID: a0ec2c1c2d58
Revises: cc8799c0f63a
Create Date: 2024-12-04 23:53:07.500982+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from ypl.db.oneoffs.initial_prompt_modifiers import add_initial_prompt_modifiers, add_simplified_prompt_modifiers, soft_remove_prompt_modifiers


# revision identifiers, used by Alembic.
revision: str = 'a0ec2c1c2d58'
down_revision: str | None = 'cc8799c0f63a'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    connection = op.get_bind()
    soft_remove_prompt_modifiers(connection)
    add_simplified_prompt_modifiers(connection)

def downgrade() -> None:
    connection = op.get_bind()
    soft_remove_prompt_modifiers(connection)
    add_initial_prompt_modifiers(connection)
