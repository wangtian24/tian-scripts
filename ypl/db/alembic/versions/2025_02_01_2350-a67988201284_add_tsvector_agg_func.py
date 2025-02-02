"""add tsvector_agg func

Revision ID: a67988201284
Revises: 495a50844300
Create Date: 2025-02-01 23:50:41.963384+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'a67988201284'
down_revision: str | None = '495a50844300'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade():
    # Add an aggregate function `tsvector_agg` that concatenates tsvectors.
    op.execute(
        """
        DO $$
        BEGIN
            -- Check if an aggregate named tsvector_agg that takes a tsvector argument exists.
            IF NOT EXISTS (
                SELECT 1
                FROM pg_aggregate a
                JOIN pg_proc p ON a.aggfnoid = p.oid
                WHERE p.proname = 'tsvector_agg'
                  AND pg_get_function_identity_arguments(p.oid) = 'tsvector'
            ) THEN
                CREATE AGGREGATE tsvector_agg(tsvector) (
                    sfunc = tsvector_concat,
                    stype = tsvector,
                    initcond = ''
                );
            END IF;
        END
        $$;
        """
    )


def downgrade():
    op.execute(
        """
        DROP AGGREGATE IF EXISTS tsvector_agg(tsvector)
        """
    )