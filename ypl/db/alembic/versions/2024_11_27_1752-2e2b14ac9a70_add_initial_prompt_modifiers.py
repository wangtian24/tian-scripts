"""add initial prompt modifiers

Revision ID: 2e2b14ac9a70
Revises: 9bc9bcd68a37
Create Date: 2024-11-27 17:52:16.401613+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from ypl.db.oneoffs.initial_prompt_modifiers import add_initial_prompt_modifiers, remove_prompt_modifiers


# revision identifiers, used by Alembic.
revision: str = '2e2b14ac9a70'
down_revision: str | None = '9bc9bcd68a37'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    add_initial_prompt_modifiers(op.get_bind())


def downgrade() -> None:
    remove_prompt_modifiers(op.get_bind())
