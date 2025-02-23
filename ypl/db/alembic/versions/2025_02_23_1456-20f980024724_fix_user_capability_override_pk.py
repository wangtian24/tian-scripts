"""Fix user_capability_override PK

Revision ID: 20f980024724
Revises: 8038ec07c962
Create Date: 2025-02-23 14:56:49.536861+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = '20f980024724'
down_revision: str | None = '8038ec07c962'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute('ALTER TABLE user_capability_overrides ADD CONSTRAINT user_capability_overrides_pk PRIMARY KEY (user_capability_override_id)')


def downgrade() -> None:
    op.execute('ALTER TABLE user_capability_overrides DROP CONSTRAINT user_capability_overrides_pk')
