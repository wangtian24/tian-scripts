"""adding a new enum called qt_eval to RewardActionEnum

Revision ID: 38eb10ab835c
Revises: a0ec2c1c2d58
Create Date: 2024-12-05 09:34:46.254790+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic_postgresql_enum import TableReference

# revision identifiers, used by Alembic.
revision: str = '38eb10ab835c'
down_revision: str | None = 'a0ec2c1c2d58'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'rewardactionenum', ['SIGN_UP', 'PROMPT', 'EVALUATION', 'QT_EVAL', 'TURN', 'FEEDBACK'],
                        [TableReference(table_schema='public', table_name='reward_action_logs', column_name='action_type')],
                        enum_values_to_rename=[])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'rewardactionenum', ['SIGN_UP', 'PROMPT', 'EVALUATION', 'TURN', 'FEEDBACK'],
                        [TableReference(table_schema='public', table_name='reward_action_logs', column_name='action_type')],
                        enum_values_to_rename=[])
    # ### end Alembic commands ###
