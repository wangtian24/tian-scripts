"""create a source instrument for plaid

Revision ID: 3aca3653b396
Revises: 13f43eabf19c
Create Date: 2024-12-28 10:53:57.987552+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from ypl.db.oneoffs.create_plaid_payment_instrument import add_plaid_payment_instrument
from ypl.db.oneoffs.create_plaid_payment_instrument import remove_plaid_payment_instrument
from dotenv import load_dotenv

# revision identifiers, used by Alembic.
revision: str = '3aca3653b396'
down_revision: str | None = '13f43eabf19c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    load_dotenv() 
    connection = op.get_bind()
    add_plaid_payment_instrument(connection)


def downgrade() -> None:
    load_dotenv() 
    connection = op.get_bind()
    remove_plaid_payment_instrument(connection)
