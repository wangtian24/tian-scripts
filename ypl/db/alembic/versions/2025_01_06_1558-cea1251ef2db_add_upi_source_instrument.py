"""Add UPI source instrument

Revision ID: cea1251ef2db
Revises: a71fbb0ab222
Create Date: 2025-01-06 15:58:54.789215+00:00

"""
from collections.abc import Sequence

from alembic import op

from ypl.db.oneoffs.create_axis_upi_payment_instrument import add_axis_upi_payment_instrument, remove_axis_upi_payment_instrument


# revision identifiers, used by Alembic.
revision: str = 'cea1251ef2db'
down_revision: str | None = 'a71fbb0ab222'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    connection = op.get_bind()
    add_axis_upi_payment_instrument(connection)


def downgrade() -> None:
    connection = op.get_bind()
    remove_axis_upi_payment_instrument(connection)
