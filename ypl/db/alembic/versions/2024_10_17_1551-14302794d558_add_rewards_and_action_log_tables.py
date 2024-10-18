"""Add Rewards and Action Log tables

Revision ID: 14302794d558
Revises: fada3bdd7de2
Create Date: 2024-10-17 15:51:30.169221+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic_postgresql_enum import TableReference
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '14302794d558'
down_revision: str | None = 'fada3bdd7de2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    sa.Enum('UNCLAIMED', 'CLAIMED', 'REJECTED', name='rewardstatusenum').create(op.get_bind())
    sa.Enum('SIGN_UP', 'PROMPT', 'EVALUATION', name='rewardactionenum').create(op.get_bind())
    op.create_table('rewards',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('modified_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('reward_id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('credit_delta', sa.Integer(), nullable=False),
    sa.Column('status', postgresql.ENUM('UNCLAIMED', 'CLAIMED', 'REJECTED', name='rewardstatusenum', create_type=False), server_default='UNCLAIMED', nullable=False),
    sa.Column('reason', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], name=op.f('fk_rewards_user_id_users')),
    sa.PrimaryKeyConstraint('reward_id', name=op.f('pk_rewards'))
    )
    op.create_index(op.f('ix_rewards_user_id'), 'rewards', ['user_id'], unique=False)
    op.create_table('reward_action_logs',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('modified_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('reward_action_log_id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('action_type', postgresql.ENUM('SIGN_UP', 'PROMPT', 'EVALUATION', name='rewardactionenum', create_type=False), nullable=False),
    sa.Column('action_details', sa.JSON(), nullable=False),
    sa.Column('associated_reward_id', sa.Uuid(), nullable=True),
    sa.ForeignKeyConstraint(['associated_reward_id'], ['rewards.reward_id'], name=op.f('fk_reward_action_logs_associated_reward_id_rewards')),
    sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], name=op.f('fk_reward_action_logs_user_id_users')),
    sa.PrimaryKeyConstraint('reward_action_log_id', name=op.f('pk_reward_action_logs'))
    )
    op.create_index(op.f('ix_reward_action_logs_user_id'), 'reward_action_logs', ['user_id'], unique=False)
    op.add_column('point_transactions', sa.Column('claimed_reward_id', sa.Uuid(), nullable=True))
    op.create_foreign_key(op.f('fk_point_transactions_claimed_reward_id_rewards'), 'point_transactions', 'rewards', ['claimed_reward_id'], ['reward_id'])
    op.sync_enum_values('public', 'pointsactionenum', ['UNKNOWN', 'SIGN_UP', 'PROMPT', 'EVALUATION', 'REWARD'],
                        [TableReference(table_schema='public', table_name='point_transactions', column_name='action_type')],
                        enum_values_to_rename=[])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'pointsactionenum', ['UNKNOWN', 'SIGN_UP', 'PROMPT', 'EVALUATION'],
                        [TableReference(table_schema='public', table_name='point_transactions', column_name='action_type')],
                        enum_values_to_rename=[])
    op.drop_constraint(op.f('fk_point_transactions_claimed_reward_id_rewards'), 'point_transactions', type_='foreignkey')
    op.drop_column('point_transactions', 'claimed_reward_id')
    op.drop_index(op.f('ix_reward_action_logs_user_id'), table_name='reward_action_logs')
    op.drop_table('reward_action_logs')
    op.drop_index(op.f('ix_rewards_user_id'), table_name='rewards')
    op.drop_table('rewards')
    sa.Enum('SIGN_UP', 'PROMPT', 'EVALUATION', name='rewardactionenum').drop(op.get_bind())
    sa.Enum('UNCLAIMED', 'CLAIMED', 'REJECTED', name='rewardstatusenum').drop(op.get_bind())
    # ### end Alembic commands ###
