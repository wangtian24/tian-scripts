"""Create yuppdb_auditor role

Revision ID: ffbcc4c27458
Revises: 20f980024724
Create Date: 2025-02-25 05:11:56.670711+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'ffbcc4c27458'
down_revision: str | None = '20f980024724'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Documenting the commands that were run manually, for future reference:
    # op.execute("CREATE ROLE yuppdb_auditor NOLOGIN")

    # op.execute("GRANT INSERT, UPDATE, DELETE ON alembic_version, capabilities, daily_account_balance_history, payment_instruments, payment_transactions, point_transactions, rewards, routing_rules, soul_role_permissions, soul_roles, soul_user_roles, user_capability_overrides TO yuppdb_auditor")
    # Only the status column of language_models is audited as there are lots of columns that get updated frequently.
    # TODO: Put those columns in a separate table, and audit everything in this table?
    # op.execute("GRANT INSERT, UPDATE (status), DELETE  ON language_models TO yuppdb_auditor")
    pass


def downgrade() -> None:
    # DO NOT DROP THE ROLE WITHOUT UNSETTING pgaudit.role
    # op.execute("REVOKE ALL ON alembic_version, capabilities, daily_account_balance_history, language_models, payment_instruments, payment_transactions, point_transactions, rewards, routing_rules, soul_role_permissions, soul_roles, soul_user_roles, user_capability_overrides FROM yuppdb_auditor")
    # op.execute("DROP ROLE yuppdb_auditor")
    pass
