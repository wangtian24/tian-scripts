"""Add Ratings table.

Revision ID: bb9efb5302a2
Revises: ea6676883cd1
Create Date: 2024-08-21 22:32:59.406112+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bb9efb5302a2'
down_revision: str | None = 'ea6676883cd1'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('ratings',
    sa.Column('rating_id', sa.Uuid(), nullable=False),
    sa.Column('model_id', sa.Uuid(), nullable=False),
    sa.Column('category_id', sa.Uuid(), nullable=False),
    sa.Column('score', sa.Float(), nullable=False),
    sa.Column('lower_bound_95', sa.Float(), nullable=False),
    sa.Column('upper_bound_95', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('modified_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['category_id'], ['categories.category_id'], name=op.f('ratings_category_id_fkey')),
    sa.ForeignKeyConstraint(['model_id'], ['language_models.model_id'], name=op.f('ratings_model_id_fkey')),
    sa.PrimaryKeyConstraint('rating_id', name=op.f('ratings_pkey')),
    sa.UniqueConstraint('model_id', 'category_id', name=op.f('uq_model_category'))
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('ratings')
    # ### end Alembic commands ###
