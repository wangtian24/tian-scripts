"""create evals table

Revision ID: bc010c576bd1
Revises: 6603078f1072
Create Date: 2024-08-16 18:35:54.195691+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'bc010c576bd1'
down_revision: str | None = '6603078f1072'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    sa.Enum('SLIDER_V0', name='evaltype').create(op.get_bind())
    op.create_table('evals',
    sa.Column('eval_id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('turn_id', sa.Uuid(), nullable=False),
    sa.Column('eval_type', postgresql.ENUM('SLIDER_V0', name='evaltype', create_type=False), nullable=False),
    sa.Column('message_1_id', sa.Uuid(), nullable=False),
    sa.Column('message_2_id', sa.Uuid(), nullable=True),
    sa.Column('score_1', sa.Float(), nullable=True),
    sa.Column('score_2', sa.Float(), nullable=True),
    sa.Column('user_comment', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('modified_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['message_1_id'], ['chat_messages.message_id'], name=op.f('evals_message_1_id_fkey')),
    sa.ForeignKeyConstraint(['message_2_id'], ['chat_messages.message_id'], name=op.f('evals_message_2_id_fkey')),
    sa.ForeignKeyConstraint(['turn_id'], ['turns.turn_id'], name=op.f('evals_turn_id_fkey')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('evals_user_id_fkey')),
    sa.PrimaryKeyConstraint('eval_id', name=op.f('evals_pkey'))
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('evals')
    sa.Enum('SLIDER_V0', name='evaltype').drop(op.get_bind())
    # ### end Alembic commands ###
