"""remove language default

Revision ID: c9f4a67e3012
Revises: 50c73502c70d
Create Date: 2024-10-09 00:06:31.755669+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = 'c9f4a67e3012'
down_revision: str | None = '50c73502c70d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

 
def upgrade() -> None:
    op.alter_column("chat_messages", "language_code", server_default=None)
    
def downgrade() -> None:
    op.alter_column("chat_messages", "language_code", server_default='EN')
