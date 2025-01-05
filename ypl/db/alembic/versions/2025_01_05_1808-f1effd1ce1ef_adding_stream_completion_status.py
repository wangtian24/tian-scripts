"""adding stream completion status 

Revision ID: f1effd1ce1ef
Revises: 33fca6b92324
Create Date: 2025-01-05 18:08:46.218719+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f1effd1ce1ef'
down_revision: str | None = '33fca6b92324'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    sa.Enum('SUCCESS', 'USER_ABORTED', 'PROVIDER_ERROR', 'SYSTEM_ERROR', name='completionstatus').create(op.get_bind())
    op.add_column('chat_messages', sa.Column('completion_status', postgresql.ENUM('SUCCESS', 'USER_ABORTED', 'PROVIDER_ERROR', 'SYSTEM_ERROR', name='completionstatus', create_type=False), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('chat_messages', 'completion_status')
    sa.Enum('SUCCESS', 'USER_ABORTED', 'PROVIDER_ERROR', 'SYSTEM_ERROR', name='completionstatus').drop(op.get_bind())
    # ### end Alembic commands ###
