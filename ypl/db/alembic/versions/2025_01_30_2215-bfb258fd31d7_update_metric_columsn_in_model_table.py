"""update metric columsn in model table

Revision ID: bfb258fd31d7
Revises: 0298ba33ba41
Create Date: 2025-01-30 22:15:22.403384+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'bfb258fd31d7'
down_revision: str | None = '0298ba33ba41'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('language_models', sa.Column('first_token_p50_latency_ms', sa.Float(), nullable=True))
    op.add_column('language_models', sa.Column('output_p50_tps', sa.Float(), nullable=True))
    op.add_column('language_models',
                  sa.Column('num_requests_in_metric_window', sa.Integer(), nullable=True))
    op.add_column('language_models',
                  sa.Column('avg_token_count', sa.Float(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    op.drop_column('language_models', 'avg_token_count')
    op.drop_column('language_models', 'num_requests_in_metric_window')
    op.drop_column('language_models', 'output_p50_tps')
    op.drop_column('language_models', 'first_token_p50_latency_ms')
    # ### end Alembic commands ###
