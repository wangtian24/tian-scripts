"""add hidden ui state

Revision ID: f846bd03e883
Revises: 41d6d2179e76
Create Date: 2024-12-27 05:47:35.235610+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic_postgresql_enum import TableReference

# revision identifiers, used by Alembic.
revision: str = 'f846bd03e883'
down_revision: str | None = '41d6d2179e76'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'messageuistatus', ['UNKNOWN', 'SEEN', 'DISMISSED', 'SELECTED', 'HIDDEN'],
                        [TableReference(table_schema='public', table_name='chat_messages', column_name='ui_status', existing_server_default="'UNKNOWN'::messageuistatus")],
                        enum_values_to_rename=[])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'messageuistatus', ['UNKNOWN', 'SEEN', 'DISMISSED', 'SELECTED'],
                        [TableReference(table_schema='public', table_name='chat_messages', column_name='ui_status', existing_server_default="'UNKNOWN'::messageuistatus")],
                        enum_values_to_rename=[])
    # ### end Alembic commands ###
