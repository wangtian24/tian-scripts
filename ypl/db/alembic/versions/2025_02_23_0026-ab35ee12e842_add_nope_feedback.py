"""add nope feedback

Revision ID: ab35ee12e842
Revises: aa18ced48acf
Create Date: 2025-02-23 00:26:48.624696+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic_postgresql_enum import TableReference

# revision identifiers, used by Alembic.
revision: str = 'ab35ee12e842'
down_revision: str | None = 'aa18ced48acf'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'rewardactionenum', ['SIGN_UP', 'EVALUATION', 'QT_EVAL', 'TURN', 'FEEDBACK', 'NOPE_FEEDBACK', 'MODEL_FEEDBACK', 'REFERRAL_BONUS_REFERRED_USER', 'REFERRAL_BONUS_REFERRER'],
                        [TableReference(table_schema='public', table_name='reward_action_logs', column_name='action_type'), TableReference(table_schema='public', table_name='reward_amount_rules', column_name='action_type', existing_server_default="'TURN'::rewardactionenum"), TableReference(table_schema='public', table_name='reward_probability_rules', column_name='action_type', existing_server_default="'TURN'::rewardactionenum")],
                        enum_values_to_rename=[])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'rewardactionenum', ['SIGN_UP', 'EVALUATION', 'QT_EVAL', 'TURN', 'FEEDBACK', 'MODEL_FEEDBACK', 'REFERRAL_BONUS_REFERRED_USER', 'REFERRAL_BONUS_REFERRER'],
                        [TableReference(table_schema='public', table_name='reward_action_logs', column_name='action_type'), TableReference(table_schema='public', table_name='reward_amount_rules', column_name='action_type', existing_server_default="'TURN'::rewardactionenum"), TableReference(table_schema='public', table_name='reward_probability_rules', column_name='action_type', existing_server_default="'TURN'::rewardactionenum")],
                        enum_values_to_rename=[])
    # ### end Alembic commands ###
