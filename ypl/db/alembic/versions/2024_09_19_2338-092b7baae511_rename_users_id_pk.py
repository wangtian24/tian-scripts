"""rename users.id pk

Revision ID: 092b7baae511
Revises: 8485cb1c12d0
Create Date: 2024-09-19 23:38:16.252845+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = '092b7baae511'
down_revision: str | None = '8485cb1c12d0'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("users", "id", new_column_name="user_id")

def downgrade() -> None:
    op.alter_column("users", "user_id", new_column_name="id")
