"""Index on name and provider_id

Revision ID: fc4ad9dd509e
Revises: c9f4a67e3012
Create Date: 2024-10-09 07:54:06.041981+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'fc4ad9dd509e'
down_revision: str | None = 'c9f4a67e3012'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('language_models_internal_name_key', 'language_models', type_='unique')
    op.drop_index('ix_language_models_name', table_name='language_models')
    op.create_index(op.f('ix_language_models_name'), 'language_models', ['name'], unique=False)
    op.create_unique_constraint(op.f('uq_language_models_internal_name'), 'language_models', ['internal_name', 'provider_id'])
    op.create_unique_constraint(op.f('uq_language_models_name'), 'language_models', ['name', 'provider_id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(op.f('uq_language_models_name'), 'language_models', type_='unique')
    op.drop_constraint(op.f('uq_language_models_internal_name'), 'language_models', type_='unique')
    op.drop_index(op.f('ix_language_models_name'), table_name='language_models')
    op.create_index('ix_language_models_name', 'language_models', ['name'], unique=True)
    op.create_unique_constraint('language_models_internal_name_key', 'language_models', ['internal_name'])
    # ### end Alembic commands ###
