"""created a system payment instrument for crypto

Revision ID: 1d82f52ceb93
Revises: ec62f1aeb675
Create Date: 2024-12-14 18:01:02.929385+00:00

"""
from collections.abc import Sequence
from dotenv import load_dotenv

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from ypl.db.oneoffs.create_crypto_payment_instrument import add_crypto_payment_instrument, remove_crypto_payment_instrument

revision: str = '1d82f52ceb93'
down_revision: str | None = 'ec62f1aeb675'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    load_dotenv() 
    connection = op.get_bind()
    add_crypto_payment_instrument(connection)


def downgrade() -> None:
    load_dotenv() 
    connection = op.get_bind()
    remove_crypto_payment_instrument(connection)
