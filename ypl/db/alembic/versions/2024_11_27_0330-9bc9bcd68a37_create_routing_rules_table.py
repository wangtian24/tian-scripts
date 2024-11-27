"""Create routing rules table

Revision ID: 9bc9bcd68a37
Revises: 984f818d04a9
Create Date: 2024-11-27 03:30:59.573715+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9bc9bcd68a37'
down_revision: str | None = '984f818d04a9'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    sa.Enum('ACCEPT', 'REJECT', 'NOOP', name='routingaction').create(op.get_bind())
    op.create_table('routing_rules',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('modified_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('routing_rule_id', sa.Uuid(), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('z_index', sa.Integer(), nullable=False),
    sa.Column('source_category', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('destination', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('target', postgresql.ENUM('ACCEPT', 'REJECT', 'NOOP', name='routingaction', create_type=False), nullable=False),
    sa.PrimaryKeyConstraint('routing_rule_id', name=op.f('pk_routing_rules')),
    sa.UniqueConstraint('source_category', 'destination', name='uq_cat_dest')
    )
    op.create_index(op.f('ix_routing_rules_destination'), 'routing_rules', ['destination'], unique=False)
    op.create_index(op.f('ix_routing_rules_source_category'), 'routing_rules', ['source_category'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_routing_rules_source_category'), table_name='routing_rules')
    op.drop_index(op.f('ix_routing_rules_destination'), table_name='routing_rules')
    op.drop_table('routing_rules')
    sa.Enum('ACCEPT', 'REJECT', 'NOOP', name='routingaction').drop(op.get_bind())
    # ### end Alembic commands ###
