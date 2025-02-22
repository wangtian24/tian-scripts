"""initialize message reviews

Revision ID: aa18ced48acf
Revises: 9504a1a871b4
Create Date: 2025-02-22 07:50:08.489287+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'aa18ced48acf'
down_revision: str | None = '9504a1a871b4'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    sa.Enum('SUCCESS', 'UNSUPPORTED', 'ERROR', name='reviewstatus').create(op.get_bind())
    sa.Enum('BINARY', 'CRITIQUE', 'SEGMENTED', name='reviewtype').create(op.get_bind())
    op.create_table('message_reviews',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('modified_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('review_id', sa.Uuid(), nullable=False),
    sa.Column('message_id', sa.Uuid(), nullable=False),
    sa.Column('review_type', postgresql.ENUM('BINARY', 'CRITIQUE', 'SEGMENTED', name='reviewtype', create_type=False), nullable=False),
    sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('reviewer_model_id', sa.Uuid(), nullable=False),
    sa.Column('status', postgresql.ENUM('SUCCESS', 'UNSUPPORTED', 'ERROR', name='reviewstatus', create_type=False), nullable=False),
    sa.ForeignKeyConstraint(['message_id'], ['chat_messages.message_id'], name=op.f('fk_message_reviews_message_id_chat_messages')),
    sa.ForeignKeyConstraint(['reviewer_model_id'], ['language_models.language_model_id'], name=op.f('fk_message_reviews_reviewer_model_id_language_models')),
    sa.PrimaryKeyConstraint('review_id', name=op.f('pk_message_reviews'))
    )
    op.create_index(op.f('ix_message_reviews_message_id'), 'message_reviews', ['message_id'], unique=False)
    op.create_index(op.f('ix_message_reviews_reviewer_model_id'), 'message_reviews', ['reviewer_model_id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_message_reviews_reviewer_model_id'), table_name='message_reviews')
    op.drop_index(op.f('ix_message_reviews_message_id'), table_name='message_reviews')
    op.drop_table('message_reviews')
    sa.Enum('BINARY', 'CRITIQUE', 'SEGMENTED', name='reviewtype').drop(op.get_bind())
    sa.Enum('SUCCESS', 'UNSUPPORTED', 'ERROR', name='reviewstatus').drop(op.get_bind())
    # ### end Alembic commands ###
