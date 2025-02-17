"""add sol doge and usdt as list of allowed crypto enums

Revision ID: 099ea2aa439a
Revises: 5fb396abb721
Create Date: 2025-02-17 06:55:41.681429+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic_postgresql_enum import TableReference

# revision identifiers, used by Alembic.
revision: str = '099ea2aa439a'
down_revision: str | None = '5fb396abb721'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'currencyenum', ['INR', 'USD', 'CBBTC', 'BTC', 'DOGE', 'ETH', 'SOL', 'USDC', 'USDT'],
                        [TableReference(table_schema='public', table_name='daily_account_balance_history', column_name='currency'), TableReference(table_schema='public', table_name='payment_transactions', column_name='currency')],
                        enum_values_to_rename=[])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'currencyenum', ['INR', 'USD', 'USDC', 'ETH', 'BTC', 'CBBTC'],
                        [TableReference(table_schema='public', table_name='daily_account_balance_history', column_name='currency'), TableReference(table_schema='public', table_name='payment_transactions', column_name='currency')],
                        enum_values_to_rename=[('SOL', 'USDC'), ('DOGE', 'USDC'), ('USDT', 'USDC')])
    # ### end Alembic commands ###
