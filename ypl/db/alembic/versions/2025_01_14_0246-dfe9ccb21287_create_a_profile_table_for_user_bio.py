"""create a profile table for user bio

Revision ID: dfe9ccb21287
Revises: f4598010fea5
Create Date: 2025-01-14 02:46:29.995868+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'dfe9ccb21287'
down_revision: str | None = 'f4598010fea5'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user_profiles',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('modified_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('user_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('educational_institution', sa.Text(), nullable=True),
    sa.Column('city', sa.Text(), nullable=True),
    sa.Column('country', sa.Text(), nullable=True),
    sa.Column('discord_username', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], name=op.f('fk_user_profiles_user_id_users')),
    sa.PrimaryKeyConstraint('user_id', name=op.f('pk_user_profiles'))
    )
    # Add ISO 3166-1 alpha-2 check constraint
    op.create_check_constraint(
        "country_iso_alpha2_check",
        "user_profiles",
        "country ~ '^[A-Z]{2}$'"
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint("country_iso_alpha2_check", "user_profiles", type_="check")
    op.drop_table('user_profiles')
    # ### end Alembic commands ###
