"""edit cross check review

Revision ID: 9baeb50def99
Revises: d58f9185263e
Create Date: 2025-03-10 23:54:57.900299+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic_postgresql_enum import TableReference

# revision identifiers, used by Alembic.
revision: str = '9baeb50def99'
down_revision: str | None = 'd58f9185263e'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'reviewtype', ['BINARY', 'CRITIQUE', 'SEGMENTED', 'NUGGETIZED', 'CROSS_CHECK_CRITIQUE', 'CROSS_CHECK_BINARY'],
                        [TableReference(table_schema='public', table_name='message_reviews', column_name='review_type')],
                        enum_values_to_rename=[('CROSS_CHECK', 'CROSS_CHECK_CRITIQUE')])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'reviewtype', ['BINARY', 'CRITIQUE', 'SEGMENTED', 'NUGGETIZED', 'CROSS_CHECK'],
                        [TableReference(table_schema='public', table_name='message_reviews', column_name='review_type')],
                        enum_values_to_rename=[('CROSS_CHECK_CRITIQUE', 'CROSS_CHECK')])
