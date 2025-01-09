"""create a table to store webhooks that partners can call

Revision ID: 472249288b96
Revises: be70aa19ea46
Create Date: 2025-01-09 15:16:00.203252+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '472249288b96'
down_revision: str | None = 'be70aa19ea46'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    sa.Enum('ACTIVE', 'INACTIVE', 'SUSPENDED', name='webhookpartnerstatusenum').create(op.get_bind())
    op.create_table('webhook_partners',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('modified_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('webhook_partner_id', sa.Uuid(), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('description', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('webhook_token', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('status', postgresql.ENUM('ACTIVE', 'INACTIVE', 'SUSPENDED', name='webhookpartnerstatusenum', create_type=False), server_default='ACTIVE', nullable=False),
    sa.Column('validation_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.PrimaryKeyConstraint('webhook_partner_id', name=op.f('pk_webhook_partners'))
    )
    op.create_index(op.f('ix_webhook_partners_name'), 'webhook_partners', ['name'], unique=False)
    op.create_index(op.f('ix_webhook_partners_webhook_token'), 'webhook_partners', ['webhook_token'], unique=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_webhook_partners_webhook_token'), table_name='webhook_partners')
    op.drop_index(op.f('ix_webhook_partners_name'), table_name='webhook_partners')
    op.drop_table('webhook_partners')
    sa.Enum('ACTIVE', 'INACTIVE', 'SUSPENDED', name='webhookpartnerstatusenum').drop(op.get_bind())
    # ### end Alembic commands ###
