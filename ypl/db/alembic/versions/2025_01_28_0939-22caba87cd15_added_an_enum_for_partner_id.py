"""added an enum for partner id

Revision ID: 22caba87cd15
Revises: 83a11cd75146
Create Date: 2025-01-28 09:39:24.457491+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic_postgresql_enum import TableReference

# revision identifiers, used by Alembic.
revision: str = '22caba87cd15'
down_revision: str | None = '83a11cd75146'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'paymentinstrumentidentifiertypeenum', ['UPI_ID', 'PHONE_NUMBER', 'EMAIL', 'CRYPTO_ADDRESS', 'BANK_ACCOUNT_NUMBER', 'PARTNER_IDENTIFIER'],
                        [TableReference(table_schema='public', table_name='payment_instruments', column_name='identifier_type')],
                        enum_values_to_rename=[])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'paymentinstrumentidentifiertypeenum', ['UPI_ID', 'PHONE_NUMBER', 'EMAIL', 'CRYPTO_ADDRESS', 'BANK_ACCOUNT_NUMBER'],
                        [TableReference(table_schema='public', table_name='payment_instruments', column_name='identifier_type')],
                        enum_values_to_rename=[])
    # ### end Alembic commands ###
