"""Update prompt modifiers

Revision ID: ac5c0652ab5d
Revises: 46da6ec47dca
Create Date: 2025-02-06 18:59:58.655703+00:00

"""
from collections.abc import Sequence
from datetime import datetime

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from ypl.db.oneoffs.update_prompt_modifiers import downgrade_modifiers, upgrade_modifiers

CUTOFF_DATE = datetime(2025, 3, 1)

# revision identifiers, used by Alembic.
revision: str = 'ac5c0652ab5d'
down_revision: str | None = '46da6ec47dca'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    connection = op.get_bind()
    if datetime.now() < CUTOFF_DATE:
        upgrade_modifiers(connection)


def downgrade() -> None:
    connection = op.get_bind()
    if datetime.now() < CUTOFF_DATE:
        downgrade_modifiers(connection)
