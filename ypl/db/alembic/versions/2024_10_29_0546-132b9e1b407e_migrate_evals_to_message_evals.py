"""migrate evals to message_evals

Revision ID: 132b9e1b407e
Revises: 7850ee216b6b
Create Date: 2024-10-29 05:46:26.379300+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from ypl.db.oneoffs.eval_messages import downgrade_eval_messages, upgrade_eval_messages

# revision identifiers, used by Alembic.
revision: str = '132b9e1b407e'
down_revision: str | None = '7850ee216b6b'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    connection = op.get_bind()
    upgrade_eval_messages(connection)


def downgrade() -> None:
    connection = op.get_bind()
    downgrade_eval_messages(connection)
