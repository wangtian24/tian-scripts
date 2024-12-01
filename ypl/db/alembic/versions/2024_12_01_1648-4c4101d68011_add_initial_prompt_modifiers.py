"""add initial prompt modifiers

Revision ID: 4c4101d68011
Revises: d882aa73c1c4
Create Date: 2024-12-01 16:48:39.549941+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from ypl.db.oneoffs.initial_prompt_modifiers import add_initial_prompt_modifiers, remove_prompt_modifiers


# revision identifiers, used by Alembic.
revision: str = '4c4101d68011'
down_revision: str | None = 'd882aa73c1c4'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    add_initial_prompt_modifiers(op.get_bind())


def downgrade() -> None:
    remove_prompt_modifiers(op.get_bind())
